#!/usr/bin/env python3
"""Stdio <-> unix-socket byte bridge for the LongWaitBench MCP daemon.

A harness's MCP client spawns THIS script as its MCP "command"; the bridge
connects to the long-lived LWB MCP daemon at the socket path given in argv[1]
and pumps newline-delimited JSON-RPC in both directions. This lets several
sequential harness processes (e.g. `opencode run --session <id>` continuations)
share ONE MCP server and therefore ONE environment state.
"""
import os
import selectors
import socket
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: mcp_bridge.py <unix-socket-path>", file=sys.stderr)
        return 2
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sys.argv[1])
    sock_file = sock.makefile("rwb", buffering=0)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sel = selectors.DefaultSelector()
    sel.register(sock_file, selectors.EVENT_READ, stdout)
    sel.register(stdin, selectors.EVENT_READ, sock_file)

    def pump(src, dst):
        # line-oriented pump: MCP JSON-RPC messages are newline-delimited,
        # and byte-count reads on pipes/sockets block until the count or EOF
        line = src.readline()
        if not line:
            return False
        if os.environ.get("LWB_BRIDGE_DEBUG"):
            tag = "in" if dst is sock_file else "out"
            sys.stderr.write(f"[bridge {tag}] {line[:200]!r}\n")
            sys.stderr.flush()
        dst.write(line)
        dst.flush()
        return True

    try:
        while True:
            for key, _ in sel.select():
                if not pump(key.fileobj, key.data):
                    return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
