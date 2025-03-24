class Counter:
    value = 1 # Class-level variable

    def increment(self):
        prev_value: int = self.value
        self.value += 1
        return prev_value
