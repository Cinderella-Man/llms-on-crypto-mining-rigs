# Rectangles

# Instructions

Count the rectangles in an ASCII diagram like the one below.

```text
   +--+
  ++  |
+-++--+
|  |  |
+--+--+
```

The above diagram contains these 6 rectangles:

```text


+-----+
|     |
+-----+
```

```text
   +--+
   |  |
   |  |
   |  |
   +--+
```

```text
   +--+
   |  |
   +--+


```

```text


   +--+
   |  |
   +--+
```

```text


+--+
|  |
+--+
```

```text

  ++
  ++


```

You may assume that the input is always a proper rectangle (i.e. the length of every line equals the length of the first line).

---

Implement this exercise in `rectangles.js`, the stub in the current directory. Keep the names of the existing functions and
classes, since the tests import them. Use only the JavaScript standard library; no packages can be installed.

`rectangles.spec.js` holds some of the tests; run them with `npm test`. A hidden test suite with more cases will be run on
your solution when you finish.
