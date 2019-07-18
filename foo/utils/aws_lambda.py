import os
import json
import boto3
import datetime

master_instance_type = 'm5.xlarge'
master_instance_volumn_sizeInGB = 64

core_instance_type = 'c5.4xlarge'
core_instance_count = 4
core_instance_volume_sizeInGB = 100

cycle_hour_1 = 1
cycle_hour_4 = 3
cycle_hour_8 = 8
cycle_hour_24 = 24


def time2string(t, fmt, minus_hour=0):
    t -= datetime.timedelta(hours=minus_hour)
    return t.strftime(fmt)


def get_steps(ActionOnFailure='TERMINATE_CLUSTER'):
    now = datetime.datetime.utcnow()
    date_start = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
    date_end = datetime.datetime.utcnow().date()
    analytics_common_args = [
        'spark-submit',
        '--master', 'yarn',
        '--deploy-mode', 'cluster',
        '--driver-cores', '1',
        '--driver-memory', '5G',
        '--executor-cores', '7',
        '--executor-memory', '8G',
        '--num-executors', '9',
        '--conf', 'spark.dynamicAllocation.enabled=true',
        '--conf', 'spark.driver.memoryOverhead=1024',
        '--conf', 'spark.yarn.executor.memoryOverhead=2048',
        '--conf', 'spark.default.parallelism=10',
        '--conf', 'spark.sql.shuffle.partitions=10'
    ]
    main_proccess_step = {
        'Name': 'main-proccess',
        'ActionOnFailure': ActionOnFailure,
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                *analytics_common_args,
                '--packages',
                'com.snowplowanalytics:snowplow-scala-analytics-sdk_2.11:0.4.1',
                '--class', 'com.vova.analytics.Main',
                's3://vomkt-emr-rec/jar/vova-bi-1.0.1.jar',
                date_start.strftime('%Y/%m/%d/%H'),
                date_end.strftime('%Y/%m/%d/%H'),
            ]
        }
    }
    intervals_24 = {
        'Name': 'page-cohort',
        'ActionOnFailure': ActionOnFailure,
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                *analytics_common_args,
                '--packages',
                'com.snowplowanalytics:snowplow-scala-analytics-sdk_2.11:0.4.1',
                '--class', 'com.vova.synchronize.OffMain',
                's3://vomkt-emr-rec/jar/vova-bi-1.2.jar',
                time2string(now, '%Y/%m/%d/%H', minus_hour=cycle_hour_24),
                time2string(now, '%Y/%m/%d/%H'),
                24
            ]
        }
    }

    intervals_8 = {
        'Name': 'page-cohort',
        'ActionOnFailure': ActionOnFailure,
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                *analytics_common_args,
                '--packages',
                'com.snowplowanalytics:snowplow-scala-analytics-sdk_2.11:0.4.1',
                '--class', 'com.vova.synchronize.OffMain',
                's3://vomkt-emr-rec/jar/vova-bi-1.2.jar',
                time2string(now, '%Y/%m/%d/%H', minus_hour=cycle_hour_8),
                time2string(now, '%Y/%m/%d/%H'),
                8
            ]
        }
    }
    intervals_4 = {
        'Name': 'page-cohort',
        'ActionOnFailure': ActionOnFailure,
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                *analytics_common_args,
                '--packages',
                'com.snowplowanalytics:snowplow-scala-analytics-sdk_2.11:0.4.1',
                '--class', 'com.vova.synchronize.OffMain',
                's3://vomkt-emr-rec/jar/vova-bi-1.2.jar',
                time2string(now, '%Y/%m/%d/%H', minus_hour=cycle_hour_4),
                time2string(now, '%Y/%m/%d/%H'),
                4
            ]
        }
    }
    steps = []
    if now.hour == 2:
        steps.append(main_proccess_step)
        steps.append(intervals_24)
    if (now.hour % 8) == 0:
        steps.append(intervals_8)
    if (now.hour % 4) == 0:
        steps.append(intervals_4)

    return steps


def run_job_flows(client, name, release_label, steps, KeepJobFlowAliveWhenNoSteps=False):
    response = client.run_job_flow(
        Name=name,
        LogUri='s3n://vomkt-emr-rec/logs/',
        ReleaseLabel=release_label,
        Instances={
            'InstanceGroups': [
                {
                    'Name': 'CORE',
                    'InstanceRole': 'CORE',
                    'InstanceType': core_instance_type,
                    'InstanceCount': core_instance_count,
                    'EbsConfiguration': {
                        'EbsBlockDeviceConfigs': [
                            {
                                'VolumeSpecification': {
                                    'VolumeType': 'gp2',
                                    'SizeInGB': core_instance_volume_sizeInGB,
                                },
                                'VolumesPerInstance': 1,
                            },
                        ],
                        'EbsOptimized': True,
                    },
                },
                {
                    'Name': 'MASTER',
                    'InstanceRole': 'MASTER',
                    'InstanceType': master_instance_type,
                    'InstanceCount': 1,
                    'EbsConfiguration': {
                        'EbsBlockDeviceConfigs': [
                            {
                                'VolumeSpecification': {
                                    'VolumeType': 'gp2',
                                    'SizeInGB': master_instance_volumn_sizeInGB,
                                },
                                'VolumesPerInstance': 1,
                            },
                        ],
                        'EbsOptimized': True,
                    },
                },
            ],
            'Ec2KeyName': 'hello-ssh',
            'KeepJobFlowAliveWhenNoSteps': KeepJobFlowAliveWhenNoSteps,
            'Ec2SubnetId': 'subnet-45aa5419',
            'EmrManagedMasterSecurityGroup': 'sg-cc3c4484',
            'EmrManagedSlaveSecurityGroup': 'sg-ec3d45a4',
            'AdditionalMasterSecurityGroups': [
                'sg-112d9159',
            ],
            'AdditionalSlaveSecurityGroups': [
                'sg-112d9159',
                'sg-cc3c4484',
            ]
        },
        Steps=steps,
        Applications=[
            {'Name': 'Hadoop'},
            {'Name': 'Spark'}
        ],
        Configurations=[
            {
                'Classification': 'spark',
                'Properties': {
                    'maximizeResourceAllocation': 'true'
                }
            },
            {
                "Classification": "capacity-scheduler",
                "Properties": {
                    "yarn.scheduler.capacity.resource-calculator": "org.apache.hadoop.yarn.util.resource.DominantResourceCalculator"
                }
            }
        ],
        VisibleToAllUsers=True,
        JobFlowRole='EMR_EC2_DefaultRole',
        ServiceRole='vomkt-emr-ArtemisEmrRole-879FPJGQM5M5',
        Tags=[
            {
                'Key': 'usage',
                'Value': 'streaming'
            },
        ],
        ScaleDownBehavior='TERMINATE_AT_TASK_COMPLETION',
        EbsRootVolumeSize=10,
    )
    return response


def add_job_flow_steps(client, cluster_id, steps):
    # for i in steps: i['ActionOnFailure'] = 'CONTINUE'
    return client.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=steps,
    )


def lambda_handler(event, context):
    print("received event: " + json.dumps(event, indent=2))
    emr_client = boto3.client('emr', 'us-east-1')
    cluster_id = None
    if not cluster_id:
        response = run_job_flows(emr_client, 'vova-bi', 'emr-5.20.0', get_steps(), False)
    else:
        response = add_job_flow_steps(emr_client, cluster_id, get_steps('CONTINUE'))
    print(response)
