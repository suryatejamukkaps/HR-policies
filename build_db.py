import shutil
from vector import DB_LOCATION, create_vector_store


if DB_LOCATION.exists():
    shutil.rmtree(DB_LOCATION)
    print("Old Chroma DB deleted.")

create_vector_store()

print("New Chroma DB created successfully.")