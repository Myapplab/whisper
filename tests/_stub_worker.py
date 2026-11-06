"""Gerçek süreçli WorkerPool testleri için model yüklemeyen işçi (test_v199)."""


def loop(name, device, ecfg, task_q, result_q, shared, load_lock):
    result_q.put({"type": "ready", "wid": name})
    while True:
        t = task_q.get()
        if t is None:
            break
        result_q.put({"type": "done", "wid": name, "tid": t["id"]})
