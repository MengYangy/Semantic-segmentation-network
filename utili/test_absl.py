from absl import app, flags, logging
'''
python ./utili/test_absl.py -help
执行上述代码，可查看帮助
'''

#设置参数，第一个是参数名称，第二个是参数默认值，无默认值可取None，第三个是参数解释
flags.DEFINE_string('str_1', 'hello world', 'Input a string.')
flags.DEFINE_integer('num_1', 0, 'Input a integer.')
flags.DEFINE_bool('bool_1', True, 'Current Status!')
flags.DEFINE_float('float_1', .8, 'Input a Float')

FLAGS = flags.FLAGS

def testFLAGS(argv=()):
    print(FLAGS.str_1)
    print(FLAGS.num_1)
    print(FLAGS.bool_1)
    print(FLAGS.float_1)


if __name__ == '__main__':
    app.run(testFLAGS)