from droidlet.tools.hitl.utils.hitl_logging import HitlLogger
import logging

hl = HitlLogger("test_logger", "1111").get_logger()
hl.error("error here")
