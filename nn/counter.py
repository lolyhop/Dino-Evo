class Counter:
    value = 0  # Class-level variable

    @classmethod
    def increment(cls):
        cls.value += 1
        return cls.value
