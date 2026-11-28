// vram_write: can the host move a card-to-card hop's second leg faster than the card does itself?
//
// On a board without peer-to-peer every hop goes card -> host memory -> card. Today the second leg is the receiving
// card's DMA reading pinned host memory (its link carries read requests out and completions back). The alternative is
// the processor writing the bytes into the receiving card's video memory through the card's BAR1 window: Vulkan's
// device-local, host-visible memory, mapped write-combined, so the link carries posted writes. This program times, on
// one card and the same sizes:
//   cpu_memcpy  memcpy from ordinary host memory into the mapped video memory, then an sfence and one read of the last
//               word written (PCIe keeps a read behind earlier posted writes, so its return means they have landed)
//   cpu_nt      the same with 16-byte non-temporal stores in place of memcpy
//   dma_h2d     cudaMemcpyAsync from pinned host memory to the card, then a stream synchronize (today's second leg)
//   dma_d2h     the reverse (today's first leg), for reference
//   cpu_read4   one 4-byte read from the mapped video memory (the round trip through the card's link), bytes ignored
// and checks after each cpu_* size that a spread of words reads back as written. The processor stays busy for the whole
// of a cpu_* copy (write-combined stores stall once the buffers fill), so its time is also the host core's time.
//
// Build (CUDA 12 toolkit and the Vulkan headers and loader):
//   g++ -O2 -std=c++17 vram_write.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64
//     -lcudart_static -ldl -lrt -lpthread -lvulkan -o vram_write
// Usage: vram_write <gpu, PCI bus order as nvidia-smi> [reps for sizes up to 64 KiB, default 300] [reps above, 20]
// Output: CSV on stdout (gpu,bus,path,bytes,reps,median_us,p10_us,p90_us,mb_per_s), notes on stderr. Exit 0, or 3 if
// a read-back was wrong.
#include <cuda_runtime.h>
#include <emmintrin.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK(x)                                                                             \
  do {                                                                                    \
    VkResult r_ = (x);                                                                    \
    if (r_ != VK_SUCCESS) {                                                               \
      fprintf(stderr, "%s:%d %s failed: VkResult %d\n", __FILE__, __LINE__, #x, (int)r_); \
      return 2;                                                                           \
    }                                                                                     \
  } while (0)
#define CU(x)                                                                                            \
  do {                                                                                                   \
    cudaError_t e_ = (x);                                                                                \
    if (e_ != cudaSuccess) {                                                                             \
      fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      return 2;                                                                                          \
    }                                                                                                    \
  } while (0)

static double now_us() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

// n a multiple of 64, both pointers 16-byte aligned
static void nt_copy(void* dst, const void* src, size_t n) {
  __m128i* d = (__m128i*)dst;
  const __m128i* s = (const __m128i*)src;
  for (size_t i = 0; i < n / 16; i += 4) {
    __m128i a = _mm_load_si128(s + i), b = _mm_load_si128(s + i + 1), c = _mm_load_si128(s + i + 2),
            e = _mm_load_si128(s + i + 3);
    _mm_stream_si128(d + i, a);
    _mm_stream_si128(d + i + 1, b);
    _mm_stream_si128(d + i + 2, c);
    _mm_stream_si128(d + i + 3, e);
  }
}

struct Stat { double med, p10, p90; };
static Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))]; };
  return {at(0.5), at(0.1), at(0.9)};
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <gpu> [reps small] [reps large]\n", argv[0]); return 1; }
  const int want = atoi(argv[1]);
  const int reps_small = argc > 2 ? atoi(argv[2]) : 300, reps_large = argc > 3 ? atoi(argv[3]) : 20;
  // one token's Q8_0 hop, a four-token verify pass, a 256-token prompt micro-batch (10,880 B a token), 16 MiB
  const std::vector<size_t> sizes = {10880, 43520, 2785280, 16u << 20};
  const size_t max_size = 16u << 20;

  // Vulkan: the card in PCI bus order, a device-local host-visible memory type (the BAR1 window)
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &ai;
  VkInstance inst;
  VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(inst, &n, pds.data());
  struct D { VkPhysicalDevice pd; uint32_t bus, dev; std::string name; };
  std::vector<D> nv;
  for (auto pd : pds) {
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    if (p2.properties.vendorID == 0x10de) nv.push_back({pd, pci.pciBus, pci.pciDevice, p2.properties.deviceName});
  }
  std::sort(nv.begin(), nv.end(), [](const D& a, const D& b) { return a.bus < b.bus; });
  if (want < 0 || want >= (int)nv.size()) { fprintf(stderr, "no NVIDIA GPU %d (%zu found)\n", want, nv.size()); return 1; }
  const D& g = nv[want];
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g.pd, &mp);
  const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((mp.memoryTypes[i].propertyFlags & need) == need) { type = i; break; }
  if (type == UINT32_MAX) { fprintf(stderr, "no device-local host-visible memory type\n"); return 1; }
  fprintf(stderr, "GPU %d %s, bus %02x: memory type %u, heap %u of %llu MiB\n", want, g.name.c_str(), g.bus, type,
          mp.memoryTypes[type].heapIndex,
          (unsigned long long)(mp.memoryHeaps[mp.memoryTypes[type].heapIndex].size >> 20));
  float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  VK(vkCreateDevice(g.pd, &dci, nullptr, &dev));
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = max_size;
  mai.memoryTypeIndex = type;
  VkDeviceMemory vmem;
  VK(vkAllocateMemory(dev, &mai, nullptr, &vmem));
  void* map = nullptr;
  VK(vkMapMemory(dev, vmem, 0, VK_WHOLE_SIZE, 0, &map));

  // CUDA: the same card by PCI bus and device number
  int ncu = 0, cud = -1;
  CU(cudaGetDeviceCount(&ncu));
  for (int i = 0; i < ncu; i++) {
    int bus = -1, devn = -1;
    cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, i);
    cudaDeviceGetAttribute(&devn, cudaDevAttrPciDeviceId, i);
    if ((uint32_t)bus == g.bus && (uint32_t)devn == g.dev) cud = i;
  }
  if (cud < 0) { fprintf(stderr, "no CUDA device on bus %02x\n", g.bus); return 1; }
  CU(cudaSetDevice(cud));
  void *dbuf = nullptr, *pinned = nullptr;
  CU(cudaMalloc(&dbuf, max_size));
  CU(cudaMallocHost(&pinned, max_size));
  cudaStream_t st;
  CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

  uint8_t* src = (uint8_t*)aligned_alloc(64, max_size);
  for (size_t i = 0; i < max_size / 4; i++) ((uint32_t*)src)[i] = (uint32_t)(i * 2654435761u);
  memcpy(pinned, src, max_size);

  printf("gpu,bus,path,bytes,reps,median_us,p10_us,p90_us,mb_per_s\n");
  auto emit = [&](const char* path, size_t bytes, std::vector<double>& t) {
    Stat s = stat(t);
    printf("%d,%02x,%s,%zu,%zu,%.2f,%.2f,%.2f,%.1f\n", want, g.bus, path, bytes, t.size(), s.med, s.p10, s.p90,
           bytes ? bytes / s.med : 0.0);
    fflush(stdout);
  };
  volatile uint32_t* vw = (volatile uint32_t*)map;
  int bad = 0;
  for (size_t sz : sizes) {
    const int reps = sz <= 65536 ? reps_small : reps_large;
    for (int mode = 0; mode < 2; mode++) {
      std::vector<double> t;
      for (int r = -3; r < reps; r++) {  // three untimed warm-ups
        const uint32_t salt = (uint32_t)(r + 7) * 0x9e3779b9u;
        ((uint32_t*)src)[sz / 4 - 1] = salt;  // the word read back changes every repetition
        double t0 = now_us();
        if (mode == 0) memcpy(map, src, sz);
        else nt_copy(map, src, sz);
        _mm_sfence();
        uint32_t last = vw[sz / 4 - 1];
        double t1 = now_us();
        if (last != salt) bad++;
        if (r >= 0) t.push_back(t1 - t0);
      }
      emit(mode == 0 ? "cpu_memcpy" : "cpu_nt", sz, t);
      for (size_t i = 0; i < sz / 4 - 1; i += std::max<size_t>(1, sz / 4 / 64))
        if (vw[i] != ((uint32_t*)src)[i]) { bad++; fprintf(stderr, "read-back wrong at word %zu of %zu\n", i, sz); break; }
    }
    for (int dir = 0; dir < 2; dir++) {
      std::vector<double> t;
      for (int r = -3; r < reps; r++) {
        double t0 = now_us();
        if (dir == 0) cudaMemcpyAsync(dbuf, pinned, sz, cudaMemcpyHostToDevice, st);
        else cudaMemcpyAsync(pinned, dbuf, sz, cudaMemcpyDeviceToHost, st);
        CU(cudaStreamSynchronize(st));
        double t1 = now_us();
        if (r >= 0) t.push_back(t1 - t0);
      }
      emit(dir == 0 ? "dma_h2d" : "dma_d2h", sz, t);
    }
  }
  {
    std::vector<double> t;
    for (int r = -3; r < 1000; r++) {
      double t0 = now_us();
      uint32_t v = vw[(r & 1023) * 16];
      double t1 = now_us();
      (void)v;
      if (r >= 0) t.push_back(t1 - t0);
    }
    emit("cpu_read4", 0, t);
  }
  fprintf(stderr, "read-backs wrong: %d\n", bad);
  vkUnmapMemory(dev, vmem);
  vkFreeMemory(dev, vmem, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  cudaFreeHost(pinned);
  cudaFree(dbuf);
  free(src);
  return bad ? 3 : 0;
}
