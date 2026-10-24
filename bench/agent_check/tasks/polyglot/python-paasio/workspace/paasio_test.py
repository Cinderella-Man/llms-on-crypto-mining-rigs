import errno
import os
import unittest
from unittest.mock import ANY, call, NonCallableMagicMock, patch

from test_utils import MockSock, MockFile, MockException, ZEN, SuperMock

from paasio import MeteredFile, MeteredSocket


class PaasioTest(unittest.TestCase):
    def test_meteredsocket_context_manager(self):
        wrapped = MockSock()
        mock = NonCallableMagicMock(wraps=wrapped, autospec=True)
        mock.__exit__.side_effect = wrapped.__exit__
        with MeteredSocket(mock) as socket:
            self.assertFalse(mock.__enter__.called)
            socket.recv(30)
        self.assertFalse(mock.__enter__.called)
        mock.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(2, len(mock.mock_calls))
        with self.assertRaisesRegex(OSError, os.strerror(errno.EBADF)):
            socket.recv(30)
        with self.assertRaisesRegex(OSError, os.strerror(errno.EBADF)):
            socket.send(b"")

    def test_meteredsocket_recv_once(self):
        mock = NonCallableMagicMock(wraps=MockSock(), autospec=True)
        with MeteredSocket(mock) as socket:
            actual_recv = socket.recv(4096)
        self.assertEqual(ZEN, actual_recv)
        self.assertEqual(1, socket.recv_ops)
        self.assertEqual(len(ZEN), socket.recv_bytes)
        self.assertEqual(1, mock.recv.call_count)

    def test_meteredsocket_recv_under_size(self):
        wrapped = MockSock(chunk=257)  # largish odd number
        mock = NonCallableMagicMock(wraps=wrapped, autospec=True)
        with MeteredSocket(mock) as socket:
            actual_recv = socket.recv(4096)
        self.assertEqual(ZEN[:257], actual_recv)
        self.assertEqual(1, socket.recv_ops)
        self.assertEqual(257, socket.recv_bytes)
        self.assertEqual(1, mock.recv.call_count)

    def test_meteredsocket_send_under_size(self):
        wrapped = MockSock(chunk=257)  # largish odd number
        mock = NonCallableMagicMock(wraps=wrapped, autospec=True)
        with MeteredSocket(mock) as socket:
            send_len = socket.send(ZEN[:123])
            self.assertEqual(ZEN[:123], wrapped._sender.getbuffer())
        self.assertEqual(123, send_len)
        self.assertEqual(1, socket.send_ops)
        self.assertEqual(123, socket.send_bytes)
        self.assertEqual(1, mock.send.call_count)

    def test_meteredsocket_stats_read_only(self):
        mock = NonCallableMagicMock(wraps=MockSock(), autospec=True)
        with MeteredSocket(mock) as socket:
            self.assertEqual(0, socket.send_ops)
            self.assertEqual(0, socket.send_bytes)
            self.assertEqual(0, socket.recv_ops)
            self.assertEqual(0, socket.recv_bytes)
            for _ in range(277):
                socket.send(b"b")
            socket.send(b"bytes")
            for _ in range(257):
                socket.recv(1)
            socket.recv(2)
            self.assertEqual(278, socket.send_ops)
            self.assertEqual(282, socket.send_bytes)
            self.assertEqual(258, socket.recv_ops)
            self.assertEqual(259, socket.recv_bytes)
            with self.assertRaises(AttributeError, msg="property 'send_ops' of 'MeteredSocket' object has no setter"):
                socket.send_ops = 0
            with self.assertRaises(AttributeError, msg="property 'send_bytes' of 'MeteredSocket' object has no setter"):
                socket.send_bytes = 0
            with self.assertRaises(AttributeError, msg="property 'recv_ops' of 'MeteredSocket' object has no setter"):
                socket.recv_ops = 0
            with self.assertRaises(AttributeError, msg="property 'recv_bytes' of 'MeteredSocket' object has no setter"):
                socket.recv_bytes = 0
            self.assertEqual(278, socket.send_ops)
            self.assertEqual(282, socket.send_bytes)
            self.assertEqual(258, socket.recv_ops)
            self.assertEqual(259, socket.recv_bytes)

    @patch("paasio.super", create=True, new_callable=SuperMock)
    def test_meteredfile_context_manager_exception_suppress(self, super_mock):
        exception = MockException("Should suppress")
        wrapped = MockFile(ZEN, exception=exception)
        mock = NonCallableMagicMock(wraps=wrapped, autospec=True)
        mock.__exit__.side_effect = wrapped.__exit__
        super_mock.mock_object = mock
        with MeteredFile() as file:
            self.assertFalse(mock.__enter__.called)
            file.read()
        self.assertFalse(mock.__enter__.called)
        mock.__exit__.assert_called_once_with(
            MockException,
            exception,
            ANY,
        )

    @patch("paasio.super", create=True, new_callable=SuperMock)
    def test_meteredfile_read_multiple(self, super_mock):
        wrapped = MockFile(ZEN)
        mock = NonCallableMagicMock(wraps=wrapped, autospec=True)
        super_mock.mock_object = mock
        actual_read = b""
        with MeteredFile() as file:
            for _ in range(5):
                actual_read += file.read(30)
        self.assertEqual(ZEN[:150], actual_read)
        self.assertEqual(5, file.read_ops)
        self.assertEqual(150, file.read_bytes)
        self.assertEqual(5, mock.read.call_count)

    @patch("paasio.super", create=True, new_callable=SuperMock)
    def test_meteredfile_write_once(self, super_mock):
        wrapped = MockFile(chunk=257)  # largish odd number
        mock = NonCallableMagicMock(wraps=wrapped, autospec=True)
        super_mock.mock_object = mock
        with MeteredFile() as file:
            write_len = file.write(ZEN)
            self.assertEqual(ZEN[:257], wrapped.getbuffer())
        self.assertEqual(257, write_len)
        self.assertEqual(1, file.write_ops)
        self.assertEqual(257, file.write_bytes)
        self.assertEqual(1, mock.write.call_count)

    @patch("paasio.super", create=True, new_callable=SuperMock)
    def test_meteredfile_stats_read_only(self, super_mock):
        mock = NonCallableMagicMock(wraps=MockFile(ZEN), autospec=True)
        super_mock.mock_object = mock
        with MeteredFile() as file:
            self.assertEqual(0, file.read_ops)
            self.assertEqual(0, file.read_bytes)
            for _ in range(57):
                file.read(1)
            file.read(2)
            self.assertEqual(58, file.read_ops)
            self.assertEqual(59, file.read_bytes)
            self.assertEqual(0, file.write_ops)
            self.assertEqual(0, file.write_bytes)
            for _ in range(77):
                file.write(b"b")
            file.write(b"bytes")
            self.assertEqual(78, file.write_ops)
            self.assertEqual(82, file.write_bytes)
            with self.assertRaises(AttributeError, msg="property 'write_ops' of 'MeteredFile' object has no setter"):
                file.write_ops = 0
            with self.assertRaises(AttributeError, msg="property 'write_bytes' of 'MeteredFile' object has no setter"):
                file.write_bytes = 0
            with self.assertRaises(AttributeError, msg="property 'read_ops' of 'MeteredFile' object has no setter"):
                file.read_ops = 0
            with self.assertRaises(AttributeError, msg="property 'read_bytes' of 'MeteredFile' object has no setter"):
                file.read_bytes = 0
            self.assertEqual(78, file.write_ops)
            self.assertEqual(82, file.write_bytes)
            self.assertEqual(58, file.read_ops)
            self.assertEqual(59, file.read_bytes)
