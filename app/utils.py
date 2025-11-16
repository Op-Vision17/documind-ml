# ml-service/app/utils.py
import os
import requests

NODE_URL = os.getenv("NODE_URL", "http://localhost:5000")
NODE_TOKEN = os.getenv("NODE_SERVICE_TOKEN", "")  # should match Node's ML_SERVICE_TOKEN

def notify_node_update(file_id: str, status: str, pages: int = None, error_message: str = None):
    """
    Notify Node backend that file ingestion status changed.
    """
    payload = {"fileId": file_id, "status": status}
    if pages is not None:
        payload["pages"] = pages
    if error_message:
        payload["errorMessage"] = error_message

    headers = {"Content-Type": "application/json"}
    if NODE_TOKEN:
        headers["x-service-token"] = NODE_TOKEN

    try:
        url = f"{NODE_URL.rstrip('/')}/api/upload/update-status"
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return True, resp.text
    except Exception as e:
        # Do not raise — just log and continue (Node might be temporarily down)
        print("notify_node_update failed:", e)
        return False, str(e)
