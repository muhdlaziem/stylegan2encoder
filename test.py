#!/usr/bin/python3

import threading
import time

class Thread (threading.Thread):
    def __init__(self, method, counter):
        threading.Thread.__init__(self)
        self.method = method
        self.counter = counter

    def run(self):
        if self.method == "A":
            printA(self.counter)
        if self.method == "B":
            printB(self.counter)
       

def printA(counter):
    while counter:
        time.sleep(2)
        print("A")
        counter -= 1

def printB(counter):
    while counter:
        time.sleep(1)
        print("B")
        counter -= 1

# threadLock = threading.Lock()
threads = []

# Create new threads
thread1 = Thread("A", 3)
thread2 = Thread("B", 5)

# Start new Threads
thread1.start()
thread2.start()

# Add threads to thread list
threads.append(thread1)
threads.append(thread2)

# Wait for all threads to complete
for t in threads:
   t.join()
print ("Exiting Main Thread")