# Example 02: Multiple Processes

In Example 01, we learned how to define and run a process. In this example we define a second.

`msetup.py`, here, defines two processes: `alice` and `bob`.

We've removed repetative configuration by moving the environment definition to a conda yaml file named `env.yml`

To launch both processes, we run the same command as before:
```sh
mrp up
```

If we wanted to run just `alice` (or just `bob`), we can do so with:
```sh
mrp up alice
```

In the code, `alice` sends data to `bob` via AlephZero, who them prints the recieved data. To see the output from `bob`
```sh
mrp logs bob
```

You can also try changing the send in `alice`, and restarting it with
```sh
mrp up -f alice
```
`bob` will start printing the new messages without restarting itself.

Finally, to stop all processes:
```sh
mrp down
```
