from multiprocessing import Manager
import _pickle as cPickle
from collections import deque, defaultdict
import numpy as np
import random


class ReplayBufferActorSide:
    def __init__(self, capacity):
        self.queue = Manager().Queue(capacity)
        self.capacity = capacity
        self.buffer = []

    def empty(self):
        return self.buffer == []

    # Actors push data into queue
    def push(self, data):
        self.queue.put(data)

    # Manager get data from queue
    def get(self):
        while not self.queue.empty():
            data = self.queue.get()
            self.buffer.append(data)

    # Send data to learner
    def send_data(self):
        send_queue = self.buffer
        self.buffer = []
        return send_queue

    def send_data_pack(self, length=4):
        """
        Split data queue into chunks to optimize transmission
        Higher-length chunks have higher risk of lossing data
        """
        ret_data_pack = []
        send_queue = self.buffer
        self.buffer = []
        for i in range(0, len(send_queue), length):
            x = i
            ret_data_pack.append(send_queue[x : x + length])
        return ret_data_pack


class ReplayBufferLearnerSide:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rcvr_buffer = []
        self.buffer = deque(maxlen=self.capacity)
        self.stats = {"sample_in": 0, "sample_out": 0, "episode_in": 0}

    def rcv_data(self, data_list):
        """
        data_list: unprocessed data from networking components
        """
        rcvd_sample_count = 0
        for data in data_list:
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            rcvd_sample_count += len(unpacked_data)
            self.buffer.extend(unpacked_data)
            self.stats["sample_in"] += len(unpacked_data)
            self.stats["episode_in"] += 1
        return rcvd_sample_count

    def rcv_data_filtered(self, data_list):
        """
        data_list: unprocessed data from networking components
        specifically filter out one-action-only actions
        """
        rcvd_sample_count = 0
        for data in data_list:
            unpacked_data_filtered = []
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            for datum in unpacked_data:
                if sum(datum["state"]["action_mask"]) > 1:
                    unpacked_data_filtered.append(datum)
            rcvd_sample_count += len(unpacked_data_filtered)
            self.buffer.extend(unpacked_data_filtered)
            self.stats["sample_in"] += len(unpacked_data_filtered)
            self.stats["episode_in"] += 1
        return rcvd_sample_count

    def sample(self, batch_size):  # only called by learner
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats["sample_out"] += batch_size
        if batch_size >= len(self.buffer):
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, batch_size)
        batch = self._pack(samples)
        return batch

    def size(self):  # Need to be called after rcv_data() call
        return len(self.buffer)

    def clear(self):  # Need to be called after rcv_data() call
        self.buffer.clear()

    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res:
                    res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)

    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)


class ReplayBufferLearnerSideTagged:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rcvr_buffer = []
        self.buffer = deque(maxlen=self.capacity)
        self.stats = {"sample_in": 0, "sample_out": 0, "episode_in": 0}
        self.tag_count = defaultdict(int)
        self.max_tag_id = 0

    def rcv_data(self, data_list):
        """
        data_list: unprocessed data from networking components
        """
        rcvd_sample_count = 0
        for data in data_list:
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            rcvd_sample_count += len(unpacked_data)
            self.buffer.extend(unpacked_data)
            self.stats["sample_in"] += len(unpacked_data)
            self.stats["episode_in"] += 1
        return rcvd_sample_count

    def rcv_data_filtered(self, data_list):
        """
        data_list: unprocessed data from networking components
        specifically filter out one-action-only actions
        """
        rcvd_sample_count = 0
        for data in data_list:
            unpacked_data_filtered = []
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            for datum in unpacked_data:
                if sum(datum["state"]["action_mask"]) > 1:
                    unpacked_data_filtered.append(datum)
                    tag_id = datum["info"][2]
                    self.tag_count[tag_id] += 1
                    if tag_id > self.max_tag_id:
                        self.max_tag_id = tag_id
            rcvd_sample_count += len(unpacked_data_filtered)
            self.buffer.extend(unpacked_data_filtered)
            self.stats["sample_in"] += len(unpacked_data_filtered)
            self.stats["episode_in"] += 1
        self.tag_count[self.max_tag_id - 2] = 0
        del self.tag_count[self.max_tag_id - 2]
        return rcvd_sample_count, self.tag_count

    def sample(self, batch_size):  # only called by learner
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats["sample_out"] += batch_size
        if batch_size >= len(self.buffer):
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, batch_size)
        batch = self._pack(samples)
        return batch

    def size(self):  # Need to be called after rcv_data() call
        return len(self.buffer)

    def clear(self):  # Need to be called after rcv_data() call
        self.buffer.clear()

    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res:
                    res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)

    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)
