
from datadog import initialize, statsd

initialize(
    statsd_host="localhost",  # important for Docker
    statsd_port=8125,
)


def increment(metric_name, tags=None):
    statsd.increment(metric_name, tags=tags or [])


def gauge(metric_name, value, tags=None):
    statsd.gauge(metric_name, value, tags=tags or [])


def timing(metric_name, value, tags=None):
    statsd.timing(metric_name, value, tags=tags or [])
