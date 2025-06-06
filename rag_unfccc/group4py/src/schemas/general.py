import json
from uuid import UUID


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle UUID objects"""
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)