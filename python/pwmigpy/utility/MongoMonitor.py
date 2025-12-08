#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 10:21:53 2025

@author: pavlis
"""
import threading
from pymongo import MongoClient
import time

class MongoMonitor(threading.Thread):
    """
    Useful class to monitor active connections to a MongoDB server.   Useful only for 
    debugging if you encourter a resource leak with MongoDB connections not being released. 
    This is a know problem if one unintentionally serializes a database client in a parallel 
    workflow.  If the mongodb server crashes with a string of "Too many open files ..." 
    messages instantiate an instance of this class to monitor the server while running 
    your application.   

    This function will probably be superceded by a more generic implementation in MsPASS.
    """
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.history = []
        self.start_time = 0
        self.daemon = True

    def run(self):
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        admin_db = client.admin
        self.start_time = time.time()
        error_count = 0
        
        while not self.stop_event.is_set():
            try:
                status = admin_db.command("serverStatus", maxTimeMS=1000)
                current_conns = status['connections']['current']
                elapsed = time.time() - self.start_time
                self.history.append({'elapsed': elapsed, 'current': current_conns})
                error_count = 0
                if len(self.history) % 10 == 0:
                    print(f"[Monitor] t={elapsed:.1f}s: connections={current_conns}")
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    print(f"[Monitor] Error: {e}")
            time.sleep(self.interval)
        
        # Close client gracefully, ignore errors during shutdown
        try:
            client.close()
        except Exception:
            pass  # Ignore errors during shutdown

    def stop(self):
        self.stop_event.set()
