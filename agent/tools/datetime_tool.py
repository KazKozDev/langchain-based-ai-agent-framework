"""
Date and time tool for temporal operations.
"""

import datetime
import pytz
from typing import Optional
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    """Input schema for datetime tool."""
    command: str = Field(description="DateTime command (now, date, time, timezone:name, format:format, add_days:N, subtract_days:N)")


class DateTimeTool:
    """Date and time operations tool."""
    
    def __init__(self):
        """Initialize datetime tool."""
        self.name = "datetime"
        self.description = (
            "Get information about time and date. "
            "Commands: "
            "'now' - current time, "
            "'date' - current date, "
            "'time' - current time, "
            "'timezone:America/New_York' - time in the specified timezone, "
            "'format:YYYY-MM-DD' - current date in the specified format, "
            "'add_days:5' - date after N days, "
            "'subtract_days:3' - date N days ago."
        )
    
    def _get_current_datetime(self, timezone: Optional[str] = None) -> datetime.datetime:
        """
        Get current datetime with optional timezone.
        
        Args:
            timezone: Timezone name (e.g., 'Europe/Moscow', 'America/New_York')
            
        Returns:
            Current datetime
        """
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                return datetime.datetime.now(tz)
            except pytz.exceptions.UnknownTimeZoneError:
                # Fallback to UTC if timezone is invalid
                return datetime.datetime.now(pytz.UTC)
        else:
            return datetime.datetime.now()
    
    def _format_datetime(self, dt: datetime.datetime, format_string: str) -> str:
        """
        Format datetime according to format string.
        
        Args:
            dt: Datetime object
            format_string: Format string
            
        Returns:
            Formatted datetime string
        """
        # Common format mappings
        format_mappings = {
            'YYYY-MM-DD': '%Y-%m-%d',
            'DD-MM-YYYY': '%d-%m-%Y',
            'MM/DD/YYYY': '%m/%d/%Y',
            'DD.MM.YYYY': '%d.%m.%Y',
            'HH:MM:SS': '%H:%M:%S',
            'HH:MM': '%H:%M',
            'YYYY-MM-DD HH:MM:SS': '%Y-%m-%d %H:%M:%S',
        }
        
        # Use mapping if available, otherwise use as-is
        python_format = format_mappings.get(format_string, format_string)
        
        try:
            return dt.strftime(python_format)
        except ValueError:
            return dt.strftime('%Y-%m-%d %H:%M:%S')  # Fallback format
    
    def _add_days(self, days: int) -> str:
        """
        Add days to current date.
        
        Args:
            days: Number of days to add
            
        Returns:
            Future date string
        """
        current_date = datetime.date.today()
        future_date = current_date + datetime.timedelta(days=days)
        return future_date.strftime('%Y-%m-%d')
    
    def _subtract_days(self, days: int) -> str:
        """
        Subtract days from current date.
        
        Args:
            days: Number of days to subtract
            
        Returns:
            Past date string
        """
        current_date = datetime.date.today()
        past_date = current_date - datetime.timedelta(days=days)
        return past_date.strftime('%Y-%m-%d')
    
    def _list_common_timezones(self) -> str:
        """List common timezones."""
        common_timezones = [
            'UTC',
            'Europe/Moscow',
            'Europe/London', 
            'America/New_York',
            'America/Los_Angeles',
            'Asia/Tokyo',
            'Asia/Shanghai',
            'Australia/Sydney',
            'Europe/Paris',
            'Europe/Berlin'
        ]
        
        result = "Commonly used timezones:\n"
        for tz_name in common_timezones:
            try:
                tz = pytz.timezone(tz_name)
                current_time = datetime.datetime.now(tz)
                result += f"  {tz_name}: {current_time.strftime('%H:%M:%S')}\n"
            except:
                continue
                
        return result
    
    def get_datetime_info(self, command: str) -> str:
        """
        Execute datetime command.
        
        Args:
            command: Command string
            
        Returns:
            Result of datetime operation
        """
        try:
            if not command or not isinstance(command, str):
                return "Error: Empty command"
            
            command = command.strip().lower()
            
            if command == "now":
                dt = self._get_current_datetime()
                return f"Current time: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
            
            elif command == "date":
                dt = self._get_current_datetime()
                return f"Current date: {dt.strftime('%Y-%m-%d')}"
            
            elif command == "time":
                dt = self._get_current_datetime()
                return f"Current time: {dt.strftime('%H:%M:%S')}"
            
            elif command.startswith("timezone:"):
                timezone = command.split(":", 1)[1].strip()
                dt = self._get_current_datetime(timezone)
                return f"Time in {timezone}: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            
            elif command.startswith("format:"):
                format_string = command.split(":", 1)[1].strip()
                dt = self._get_current_datetime()
                formatted = self._format_datetime(dt, format_string)
                return f"Current date/time in format '{format_string}': {formatted}"
            
            elif command.startswith("add_days:"):
                try:
                    days = int(command.split(":", 1)[1].strip())
                    future_date = self._add_days(days)
                    return f"Date in {days} days: {future_date}"
                except ValueError:
                    return "Error: Invalid number of days"
            
            elif command.startswith("subtract_days:"):
                try:
                    days = int(command.split(":", 1)[1].strip())
                    past_date = self._subtract_days(days)
                    return f"Date {days} days ago: {past_date}"
                except ValueError:
                    return "Error: Invalid number of days"
            
            elif command == "timezones":
                return self._list_common_timezones()
            
            else:
                return (
                    "Available commands:\n"
                    "- now - current time\n"
                    "- date - current date\n"
                    "- time - current time\n"
                    "- timezone:<zone_name> - time in the timezone\n"
                    "- format:<format> - date in the specified format\n"
                    "- add_days:N - date after N days\n"
                    "- subtract_days:N - date N days ago\n"
                    "- timezones - list of timezones"
                )
                
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def get_tool(self) -> StructuredTool:
        """Get LangChain StructuredTool instance.""" 
        return StructuredTool.from_function(
            func=self.get_datetime_info,
            name=self.name,
            description=self.description,
            args_schema=DateTimeInput
        )