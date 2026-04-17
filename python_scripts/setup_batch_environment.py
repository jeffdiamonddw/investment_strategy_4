import boto3
import time
import json

ACCOUNT_ID = "129861351772"
REGION = "us-west-2" 
SUBNET_ID = "subnet-26bc127f" 
SG_ID = "sg-e51be181" 
ENV_NAME = "TripleThreat-Env-V4" # Versioned to V3
QUEUE_NAME = "TripleThreat-Queue-V4"
ROLE_NAME = "TripleThreatWorkerRole"

batch = boto3.client('batch', region_name=REGION)
iam = boto3.client('iam')

def ensure_iam_setup():
    print("--- Configuring Unified IAM Permissions ---")
    
    # Combined Trust Policy for EC2, ECS, and Batch
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": ["ec2.amazonaws.com", "ecs-tasks.amazonaws.com", "batch.amazonaws.com"]
            },
            "Action": "sts:AssumeRole"
        }]
    }

    # 1. Ensure Worker Role
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust_policy))
        print(f"Created Role: {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=json.dumps(trust_policy))
        print(f"Updated Trust Policy for {ROLE_NAME}")

    # 2. Attach Managed Policies
    policies = [
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly',
        'arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role',
        'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
    ]
    for policy in policies:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy)

    # 3. Add the CRITICAL PassRole Inline Policy
    pass_role_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": [
                f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}",
                f"arn:aws:iam::{ACCOUNT_ID}:role/ecsTaskExecutionRole"
            ]
        }]
    }
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="BatchPassRolePolicy", PolicyDocument=json.dumps(pass_role_policy))
    print("Attached PassRole Permissions.")

    # 4. Instance Profile
    try:
        iam.create_instance_profile(InstanceProfileName=ROLE_NAME)
    except iam.exceptions.EntityAlreadyExistsException: pass
    try:
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME)
    except Exception: pass

def setup_infrastructure():
    print(f"\n--- Launching {ENV_NAME} (Test Mode) ---")

    # This fixes the 'spotIamFleetRole is required' error
    spot_fleet_role = f'arn:aws:iam::{ACCOUNT_ID}:role/aws-service-role/spotfleet.amazonaws.com/AWSServiceRoleForEC2SpotFleet'

    try:
        batch.create_compute_environment(
            computeEnvironmentName=ENV_NAME,
            type='MANAGED',
            state='ENABLED',
            computeResources={
                'type': 'SPOT',
                'maxvCpus': 96,
                'minvCpus': 0,
                'instanceTypes': ['c8g.24xlarge'], 
                'subnets': [SUBNET_ID],
                'securityGroupIds': [SG_ID],
                'instanceRole': f'arn:aws:iam::{ACCOUNT_ID}:instance-profile/{ROLE_NAME}',
                'spotIamFleetRole': spot_fleet_role 
            },
            serviceRole=f'arn:aws:iam::{ACCOUNT_ID}:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch'
        )
    except Exception as e: 
        print(f"CE Creation Note: {e}")

    # Wait for VALID status
    while True:
        resp = batch.describe_compute_environments(computeEnvironments=[ENV_NAME])
        if resp['computeEnvironments']:
            ce = resp['computeEnvironments'][0]
            status = ce['status']
            print(f"Status: {status}")
            if status == 'VALID': break
            if status == 'INVALID':
                print(f"Error: {ce.get('statusReason')}")
                return
        time.sleep(15)

    # 3. Create Queue
    try:
        batch.create_job_queue(
            jobQueueName=QUEUE_NAME,
            state='ENABLED',
            priority=1,
            computeEnvironmentOrder=[{'order': 1, 'computeEnvironment': ENV_NAME}]
        )
        print(f"Queue {QUEUE_NAME}: Online")
    except Exception as e:
        print(f"Queue Note: {e}")

    # 4. Register Test Job Definition (Scaled down to match the .large instance)
    batch.register_job_definition(
        jobDefinitionName='TripleThreat-Optimization-V4', # New Name
        type='container',
        containerProperties={
            'image': f'{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/optimization:latest',
            'vcpus': 94,          # UPDATED: Use the high-core count
            'memory': 180000,     # UPDATED: Give it ~180GB RAM
            'jobRoleArn': f'arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}',
            'executionRoleArn': f'arn:aws:iam::{ACCOUNT_ID}:role/ecsTaskExecutionRole'
        }
    )
    print("Test Job Definition: Registered")
    print(f"\n--- TEST ENVIRONMENT READY ---")

if __name__ == "__main__":
    ensure_iam_setup()
    setup_infrastructure()