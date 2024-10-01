import os
from typing import Tuple, Optional

import numpy as np
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

from metrics.group_diversity import group_diversity

load_dotenv()
DIVERSITY_THRESHOLD = 0.5
EMBEDDINGS_PATH = os.getenv("embeddings_path")
UPDATE_INTERVAL = int(os.getenv("update_interval", 10))

async def periodic_load_embeddings():
    while True:
        load_embeddings()
        await asyncio.sleep(UPDATE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):

    task = asyncio.create_task(periodic_load_embeddings())
    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
embeddings = {}


def load_embeddings() -> None:
    """Load embeddings from file."""
    global embeddings
    path = os.path.join(
        os.path.dirname(__file__), EMBEDDINGS_PATH
    )
    embeddings_raw = np.load(path, allow_pickle=True)

    print(f"Loaded embeddings shape: {len(embeddings_raw)}")
    print(f"Sample embedding: {next(iter(embeddings_raw.values()))}")

    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding
    print("Embeddings updated successfully")

    return {}


@app.get("/diversity/")
def diversity(item_ids: str,
              diversity_metric: Optional[str] = 'kde',
              num_neighbors: Optional[int] = 5
              ) -> Tuple[bool, float]:
    """Calculate group diversity of items
    and reject it if it's too low using KDE (by default) or KNN"""

    item_ids = [int(item) for item in item_ids.split(",")]

    item_embeddings = []
    for item_id in item_ids:
        if item_id not in embeddings:
            print(f"Item ID {item_id} not found in embeddings")
            continue
        item_embeddings.append(embeddings[item_id])

    if not item_embeddings:
        print("No valid item embeddings found")
        return

    reject, diversity = group_diversity(
        np.array(item_embeddings),
        DIVERSITY_THRESHOLD,
        diversity_metric=diversity_metric,
        num_neighbors=num_neighbors
        )

    return bool(reject), float(diversity)


def main() -> None:
    """Run application"""
    uvicorn.run("filtering_service:app", host="localhost")


if __name__ == "__main__":
    main()
