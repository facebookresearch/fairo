from droidlet.parallel import BackgroundTask


class Foo:
    def __init__(self):

        def init_fn(a):
            def bar(b):
                return 2 + b
            return [bar]

        def process_fn(f, b, c):
            f = f[0]
            x = f(b) + 5
            print(x)
            return x, x
    
        b = BackgroundTask(init_fn=init_fn, init_args=(2,), process_fn=process_fn)
        b.start()
        self.b = b

    def forward(self):        
        self.b.put(5, 6)
        print("here:", self.b.get(timeout=2))
    

if __name__ == "__main__":
    foo = Foo()
    foo.forward()
