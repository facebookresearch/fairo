# Example 03: API

Here, we see an example of how to load a prebuilt docker image.

The alephzero/api image we load is an AlephZero web-bridge.

MRP relays process stdout/stderr through AlephZero topics, allowing us to build pure-client side log monitoring.

To launch all the processes, we run the same command as before:
```sh
mrp up
```

Now open your browser to `file:///path/to/mrp/examples/03_api/stdout.html`

(make sure to update the path to the actual filepath on your machine)

You should see the print/log statements from `alice` and `bob`.

Finally, to stop all processes:
```sh
mrp down
```
