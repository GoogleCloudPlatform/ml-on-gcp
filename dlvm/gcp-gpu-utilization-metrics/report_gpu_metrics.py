# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Report GPU metrics.

Installs a monitoring agent that monitors the GPU usage on the instance.
This will auto create the GPU metrics.
"""
from enum import Enum

import argparse
import csv
import subprocess
import time
import requests

from google.cloud import monitoring_v3

METADATA_SERVER = 'http://metadata/computeMetadata/v1/instance/'
METADATA_FLAVOR = {'Metadata-Flavor': 'Google'}


class GpuMetrics(Enum):
    """Types of metrics."""

    TIMESTAMP = 'timestamp'
    NAME = 'name'
    PCI_BUS_ID = 'pci.bus_id'
    DRIVER_VERSION = 'driver_version'
    PSTATE = 'pstate'
    PCIE_LINK_GEN_MAX = 'pcie.link.gen.max'
    PCIE_LINK_GEN_CURRENT = 'pcie.link.gen.current'
    TEMPERATURE_GPU = 'temperature.gpu'
    UTILIZATION_GPU = 'utilization.gpu'
    UTILIZATION_MEMORY = 'utilization.memory'
    MEMORY_TOTAL = 'memory.total'
    MEMORY_FREE = 'memory.free'
    MEMORY_USED = 'memory.used'

    @classmethod
    def all(cls):
        return (
            cls.TIMESTAMP,
            cls.NAME,
            cls.PCI_BUS_ID,
            cls.PSTATE,
            cls.PCIE_LINK_GEN_MAX,
            cls.PCIE_LINK_GEN_CURRENT,
            cls.TEMPERATURE_GPU,
            cls.UTILIZATION_GPU,
            cls.UTILIZATION_MEMORY,
            cls.MEMORY_TOTAL,
            cls.MEMORY_FREE,
            cls.MEMORY_USED)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid metric key provided: %s.' % key)


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sleep',
        type=int,
        default=15,
        help='number of seconds to wait while collecting metrics, default=15')
    args, _ = parser.parse_known_args()
    return args


def report_metric(value, metric_type, resource_values):
    """Create time series for report.

    Args:
      value: (int) Report metric value.
      metric_type: (str) Metric type
      resource_values: (dict) Contains resources information
    """
    client = resource_values.get('client')
    project_id = resource_values.get('project_id')
    instance_id = resource_values.get('instance_id')
    zone = resource_values.get('zone')

    project_name = client.common_project_path(project_id)
    # TimeSeries definition.
    series = monitoring_v3.types.TimeSeries()
    series.metric.type = 'custom.googleapis.com/{type}'.format(type=metric_type)
    series.resource.type = 'gce_instance'
    series.resource.labels['instance_id'] = instance_id
    series.resource.labels['zone'] = zone
    series.resource.labels['project_id'] = project_id
    point = series.points.add()
    point.value.int64_value = value
    now = time.time()
    point.interval.end_time.seconds = int(now)
    point.interval.end_time.nanos = int(
        (now - point.interval.end_time.seconds) * 10 ** 9)
    client.create_time_series(project_name, [series])


def get_nvidia_smi_utilization(gpu_query_metric):
    """Obtain NVIDIA SMI utilization.

    Args:
      gpu_query_metric: (str) GPU query name.

    Returns:
      An `int` of smi utilization. Average in file
    """
    csv_file_path = '/tmp/nvidia_smi_metrics.csv'
    lines = 0
    usage = 0
    subprocess.check_call([
        '/bin/bash', '-c',
        'nvidia-smi --query-gpu={gpu_query_metric} -u --format=csv'
        ' > {csv_file_path}'.format(
            gpu_query_metric=gpu_query_metric.value,
            csv_file_path=csv_file_path)
    ])
    with open(csv_file_path) as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        for row in rows:
            lines += 1
            if lines > 1:
                usage += int(row[0])
    # Calculate average
    return int(usage / (lines - 1))


def get_metric_value(metric_name=''):
    """Supported metric names:
      timestamp
      name
      pci.bus_id
      driver_version
      pstate
      pcie.link.gen.max
      temperature.gpu
      utilization.gpu
      utilization.memory
      memory.total
      memory.free
      memory.used

    https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia
    -smi-queries

    Args:
      metric_name: (str) Metric name

    Returns:
      An `int` of smi utilization.
    """
    return get_nvidia_smi_utilization(metric_name)


def report_metrics(resource_values, sleep_time, metrics):
    """Collects metrics

    Args:
      resource_values:(dict) Dict to pass to Stackdriver.
      sleep_time:(int) Wait time.
      metrics:(dict) Metrics to collect.

    Returns:
    """
    while True:
        report_metric(
            value=get_metric_value(metrics.get('utilization_memory')),
            metric_type='utilization_memory',
            resource_values=resource_values)
        report_metric(
            value=get_metric_value(metrics.get('utilization_gpu')),
            metric_type='utilization_gpu',
            resource_values=resource_values)
        report_metric(
            value=get_metric_value(metrics.get('memory_used')),
            metric_type='memory_used',
            resource_values=resource_values)
        time.sleep(sleep_time)


def _get_resource_values():
    """Get Resources Values

    :return:
    """
    # Get instance information
    data = requests.get('{}zone'.format(METADATA_SERVER),
                        headers=METADATA_FLAVOR).text
    instance_id = requests.get(
        '{}id'.format(METADATA_SERVER), headers=METADATA_FLAVOR).text
    client = monitoring_v3.MetricServiceClient()
    # Collect zone
    zone = data.split('/')[3]
    # Collect project id
    project_id = data.split('/')[1]
    resource_values = {
        'client': client,
        'instance_id': instance_id,
        'zone': zone,
        'project_id': project_id
    }
    return resource_values


def main(args):
    resource_values = _get_resource_values()
    # Dictionary with default metrics.
    metrics = {
        'utilization_memory': GpuMetrics.UTILIZATION_MEMORY,
        'utilization_gpu': GpuMetrics.UTILIZATION_GPU,
        'memory_used': GpuMetrics.MEMORY_USED
    }
    report_metrics(resource_values, args.sleep, metrics)


if __name__ == '__main__':
    args = get_args()
    main(args)
