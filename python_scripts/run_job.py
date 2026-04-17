import boto3
import sys

# Initialize the Batch client for the Oregon region
batch = boto3.client('batch', region_name='us-west-2')

def launch_optimization(job_name):
    print(f"--- Submitting '{job_name}' to the V4 96-core Fleet ---")
    
    try:
        # Ensure these match the V3 setup precisely
        response = batch.submit_job(
            jobName='Production_Final_Push_V15',
            jobQueue='TripleThreat-Queue-V4',
            jobDefinition='TripleThreat-Optimization-V4',
            retryStrategy={'attempts': 10} 
        )

        
        print(f"Success! Job submitted to us-west-2.")
        print(f"Job ID: {response['jobId']}")
        print(f"Monitor here: https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2#jobs/detail/{response['jobId']}")
        
    except Exception as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    # Use the first argument as the job name, or default to 'TripleThreat_Run_V2'
    name = sys.argv[1] if len(sys.argv) > 1 else "TripleThreat_Run_V4"
    launch_optimization(name)