# Troubleshooting blog generator connectivity

## Connection timeouts to the blog server
- **Symptom:** Client requests fail with an error such as:
  - `HTTPConnectionPool(host='192.168.40.91', port=8000): Read timed out. (read timeout=600)`
- **Meaning:** The HTTP client waited 600 seconds for a response from the blog server but nothing arrived. This usually indicates that the blog server at `192.168.40.91:8000` is not reachable, not running, or stuck processing the request.

### How to resolve
1. Ensure the blog server is running and reachable:
   - On the server machine, start it with `python apps/scripts/blog_server.py` (the server listens on port `8000` by default).
   - Confirm the terminal shows `Serving blog generator on http://0.0.0.0:8000/api/generate_blog`.
2. Check network connectivity from the client to `192.168.40.91:8000` (e.g., VPN, firewall, or port-forwarding settings).
3. If the server is running but the request still hangs, inspect the server logs for long-running generation or errors in the blog builder.

### Tip
To avoid long waits during failures, configure your HTTP client with a shorter timeout and display a user-friendly message when the server is unreachable.
