from multiprocessing import Process,Queue
import time
def foo1(x,queue):
    for i in range(x):
        time.sleep(2)
        print("Foo1",i)
        queue.put(i,block=False)
    queue.put(None)

def foo2(queue):
    while True:
        try:
            item = queue.get(timeout=0.5)
            print("Foo2",item)
        except :
            print('Consumer: gave up waiting...', flush=True)
            continue

    
if __name__ == "__main__":
    q=Queue(maxsize=100)
    t1=Process(target=foo1, args=(10,q))
    t2=Process(target=foo2, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
 
