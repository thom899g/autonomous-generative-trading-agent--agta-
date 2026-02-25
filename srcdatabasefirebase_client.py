"""
Firebase client for AGTA system.
Handles Firestore database operations and real-time streaming.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from datetime import datetime, timedelta
from functools import wraps
import threading

import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud import firestore as google_firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from ..config.settings import settings