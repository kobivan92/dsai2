#!/usr/bin/env python3
"""
Auto Report Generator
Automatically generates Global Explanations reports when files are uploaded
"""

import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from generate_global_explanations_report import analyze_dataset

class ReportGenerator(FileSystemEventHandler):
    def __init__(self, uploads_dir="uploads", reports_dir="reports"):
        self.uploads_dir = uploads_dir
        self.reports_dir = reports_dir
        self.processed_files = set()
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            filepath = event.src_path
            if filepath not in self.processed_files:
                self.processed_files.add(filepath)
                print(f"\n🔄 New CSV file detected: {filepath}")
                
                # Wait a moment for file to be fully written
                time.sleep(2)
                
                try:
                    # Generate report in a separate thread to avoid blocking
                    thread = threading.Thread(target=self.generate_report, args=(filepath,))
                    thread.daemon = True
                    thread.start()
                except Exception as e:
                    print(f"❌ Error processing file: {str(e)}")
    
    def generate_report(self, filepath):
        try:
            print(f"📊 Generating SHAP Global Explanations report for: {os.path.basename(filepath)}")
            report_path = analyze_dataset(filepath, n_rows=1000)  # Limit to 1000 rows for speed
            print(f"✅ SHAP Report generated: {report_path}")
            
            # Try to open the report
            try:
                import webbrowser
                webbrowser.open(f'file://{os.path.abspath(report_path)}')
                print("🌐 SHAP Report opened in browser")
            except:
                print(f"💡 Open this file in your browser: {report_path}")
                
        except Exception as e:
            print(f"❌ Failed to generate SHAP report: {str(e)}")

def start_monitoring(uploads_dir="uploads", reports_dir="reports"):
    """Start monitoring the uploads directory for new CSV files"""
    
    # Create directories if they don't exist
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    print(f"🔍 Starting file monitoring...")
    print(f"📁 Monitoring directory: {uploads_dir}")
    print(f"📄 Reports will be saved to: {reports_dir}")
    print(f"⏳ Waiting for CSV files to be uploaded...")
    print(f"💡 Upload a CSV file to {uploads_dir} to generate a report automatically")
    
    event_handler = ReportGenerator(uploads_dir, reports_dir)
    observer = Observer()
    observer.schedule(event_handler, uploads_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Monitoring stopped")
    
    observer.join()

if __name__ == "__main__":
    start_monitoring() 