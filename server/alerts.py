"""
Alert system for ROS Launchpad
"""

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


async def send_process_failure_alert(
    config: dict,
    process_name: str
) -> None:
    """Send alert when a process fails"""
    if not config:
        logger.warning("No config loaded, cannot send alerts")
        return

    alerts = config.get('alerts', [])
    deployment = config.get('name', 'unknown')

    if not alerts:
        logger.info("No alerts configured")
        return

    for alert in alerts:
        if alert.get('type') == 'slack' and alert.get('url'):
            try:
                deploy_str = f"Deployment: _{deployment}_"
                message = {
                    'text': f'*Process failed*\n - {deploy_str}\n - Process: _{process_name}_'
                }

                urllib.request.urlopen(
                    alert['url'],
                    json.dumps(message).encode()
                )

                logger.info("Alert sent for %s", process_name)
            except (urllib.error.URLError, OSError, ValueError) as e:
                logger.error("Failed to send alert: %s", e)


async def send_test_alert(config: dict) -> None:
    """Send a test alert"""
    if not config:
        logger.warning("No config loaded, cannot send alerts")
        return

    alerts = config.get('alerts', [])
    deployment = config.get('name', 'unknown')

    if not alerts:
        logger.info("No alerts configured")
        return

    for alert in alerts:
        if alert.get('type') == 'slack' and alert.get('url'):
            try:
                deploy_str = f"Deployment: _{deployment}_"
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = {
                    'text': f'🧪 *Alert Test*\n - {deploy_str}\n - Test Time: {time_str}'
                }

                urllib.request.urlopen(
                    alert['url'],
                    json.dumps(message).encode()
                )

                logger.info("Test alert sent successfully")
            except (urllib.error.URLError, OSError, ValueError) as e:
                logger.error("Failed to send alert: %s", e)


async def test_alert_system(config: Optional[dict]) -> dict:
    """Test the alert system and return result"""
    if not config:
        return {"success": False, "message": "No config loaded"}

    alerts = config.get('alerts', [])
    if not alerts:
        return {"success": False, "message": "No alerts configured"}

    try:
        await send_test_alert(config)
        return {"success": True, "message": "Test alert sent successfully"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"success": False, "message": f"Failed to send test alert: {str(e)}"}
