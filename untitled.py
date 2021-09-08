import tensorflow as tf

'''w = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[2., 2., 2.]]

print(w, b)

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)
    
[dl_dw, dl_db] = tape.gradient(loss, [w, b])

print(dl_dw)
print(dl_db)'''


strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
@tf.function
def run():
    def value_fn(value_context):
        return 3#value_context.num_replicas_in_sync
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
          value_fn))
    def replica_fn2(input):
        return input*2
    return strategy.run(replica_fn2, args=(distributed_values,))
result = run()

print(result.numpy())