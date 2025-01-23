from multiprocessing.shared_memory import SharedMemory, ShareableList
import _pickle as cPickle
import time


class ModelPoolServer:

    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 1024
        self.shared_model_list = ShareableList(
            [" " * metadata_size] * capacity + [self.n], name=name
        )

    def push(self, state_dict, metadata={}):
        n = self.n % self.capacity
        if self.model_list[n]:
            # FIFO: release shared memory of older model
            self.model_list[n]["memory"].unlink()

        data = cPickle.dumps(state_dict)  # model parameters serialized to bytes
        memory = SharedMemory(create=True, size=len(data))
        memory.buf[:] = data[:]
        # print('Created model', self.n, 'in shared memory', memory.name)

        metadata = metadata.copy()
        metadata["_addr"] = memory.name
        metadata["id"] = self.n
        metadata["memory"] = memory
        self.model_list[n] = metadata
        self.shared_model_list[n] = cPickle.dumps(metadata)
        self.n += 1
        self.shared_model_list[-1] = self.n

    def cleanup(self):
        for i in range(self.capacity):
            if (
                isinstance(self.model_list[i], dict)
                and self.model_list[i].get("memory", "") != ""
            ):
                self.model_list[i]["memory"].unlink()
        self.shared_model_list.shm.close()
        self.shared_model_list.shm.unlink()

    # def __del__(self):
    #     for i in range(self.capacity):
    #         if (
    #             isinstance(self.model_list[i], dict)
    #             and self.model_list[i].get("memory", "") != ""
    #         ):
    #             self.model_list[i]["memory"].unlink()
    #     self.shared_model_list.shm.close()
    #     self.shared_model_list.shm.unlink()


class ModelPoolClient:

    def __init__(self, name):
        while True:
            try:
                self.shared_model_list = ShareableList(name=name)
                break
            except:
                time.sleep(0.5)
        self.capacity = len(self.shared_model_list) - 1
        self.model_list = [None] * self.capacity
        self.n = 0
        self._update_model_list()

    def _update_model_list(self):
        time.sleep(0.05)
        n = self.shared_model_list[-1]
        if n > self.n:
            # new models available, update local list
            for i in range(max(self.n, n - self.capacity), n):
                self.model_list[i % self.capacity] = cPickle.loads(
                    self.shared_model_list[i % self.capacity]
                )
            self.n = n

    def get_model_list(self):
        self._update_model_list()
        model_list = []
        if self.n >= self.capacity:
            model_list.extend(self.model_list[self.n % self.capacity :])
        model_list.extend(self.model_list[: self.n % self.capacity])
        return model_list

    def get_latest_model(self):
        self._update_model_list()
        while self.n == 0:
            time.sleep(1)
            self._update_model_list()
        return self.model_list[(self.n + self.capacity - 1) % self.capacity]

    def load_model(self, metadata):
        self._update_model_list()
        time.sleep(0.05)
        n = metadata["id"]
        if n < self.n - self.capacity:
            return None
        memory = SharedMemory(name=metadata["_addr"])
        state_dict = cPickle.loads(memory.buf)
        return state_dict
