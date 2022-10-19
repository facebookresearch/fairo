import redis

print("============")
print("foo =", redis.Redis().get("foo"))
print("============")
