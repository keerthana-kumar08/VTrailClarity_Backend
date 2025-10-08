import asyncio
from typing import Dict, List



class QueueManager:
    def __init__(self):
        self.message_queue: Dict[str, List[Dict]] = {}
        self.secondary_queue: Dict[str, List[Dict]] = {}
        self.queue_lock = asyncio.Lock()

    async def add_message(self, trial_id: int, message_data: Dict):
        async with self.queue_lock:
            if trial_id not in self.message_queue:
                self.message_queue[trial_id] = []
            self.message_queue[trial_id].append({"trial_id": trial_id, "message": message_data})

    async def swap_queues(self):
        async with self.queue_lock:
            if self.message_queue and not self.secondary_queue:
                self.message_queue, self.secondary_queue = self.secondary_queue, self.message_queue
            elif self.secondary_queue and self.message_queue:
                for key in self.message_queue.keys():
                    if key not in self.secondary_queue:
                        self.secondary_queue[key] = []
                    self.secondary_queue[key].extend(self.message_queue.get(key, []))
                self.message_queue.clear()

    async def clear_secondary_queue(self):
        async with self.queue_lock:
            self.secondary_queue.clear()

    async def get_queues(self):
        async with self.queue_lock:
            return dict(self.message_queue), dict(self.secondary_queue)

queue_manager = QueueManager()