import os
import shutil
from datetime import datetime, timedelta
import pytz

root_dir = 'folsom_images/'  # path to the directory containing the month directories
destination_dir = 'new_folsom_images/'  # path to the directory where you want to copy the images

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Define the timezones
utc_tz = pytz.timezone('UTC')
la_tz = pytz.timezone('America/Los_Angeles')

# Define the exact minutes we're interested in
# exact_minutes = {0, 15, 30, 45}
# # Define the minutes where we consider 59 seconds before the next minute
# previous_minutes = {14, 29, 44, 59}

# Walk through the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Process only the leaf directories (those containing the images)
    print("Walking")
    if not dirnames:
        for filename in filenames:
            # Ignore non-jpg files
            if not filename.endswith('.jpg'):
                continue
            
            dt_str = filename.split('_')  # get the date and time string
            dt_str = (''.join(dt_str))[:-4]
            # print(dt_str)
            dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
            
            # If the minute isn't in exact_minutes or (in previous_minutes and the second is 59), skip this file
            # if not ((dt.minute in exact_minutes and dt.second == 0) or (dt.minute in previous_minutes and dt.second == 59)):
            #     continue

            # if dt.minute in previous_minutes and dt.second == 59:
            #     dt = dt + timedelta(seconds=1)
            
            # print(dt.minute, dt.second)
            # Convert the timezone
            dt = dt.replace(tzinfo=utc_tz)
            dt = dt.astimezone(la_tz)
            
            # Form the new filename
            new_filename = dt.strftime('%Y%m%d_%H%M%S.jpg')
            
            # Copy and rename the file
            shutil.copy(os.path.join(dirpath, filename), os.path.join(destination_dir, new_filename))

