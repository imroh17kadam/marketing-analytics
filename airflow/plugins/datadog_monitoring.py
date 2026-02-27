import time

from datadog import initialize, statsd

# IMPORTANT: Airflow runs inside Docker container
initialize(statsd_host="host.docker.internal", statsd_port=8125)


def track_task(metric_prefix):

    class Tracker:

        def __init__(self):
            self.start_time = None

        def start(self):
            self.start_time = time.time()
            statsd.increment(f"{metric_prefix}.started")

        def success(self):
            statsd.increment(f"{metric_prefix}.completed")

        def fail(self):
            statsd.increment(f"{metric_prefix}.failed")

        def duration(self):
            if self.start_time:
                statsd.timing(
                    f"{metric_prefix}.duration", time.time() - self.start_time
                )

    return Tracker()
