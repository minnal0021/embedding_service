"""Unit tests for the gateway's document and query lanes, against a fake
llama-server. Run: uv run python -m unittest discover tests"""

import asyncio
import json
import os
import sys
import unittest

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("EMBEDDING_MODELS", "gemma")

from server import SERVERS, DynamicBatcher, ModelServerError  # noqa: E402

SERVER = next(iter(SERVERS.values()))


def _vectors(inputs: list[str]) -> httpx.Response:
    # Each text "N" embeds as [N, 1], so results can be matched to inputs.
    data = [{"index": i, "embedding": [float(t), 1.0]} for i, t in enumerate(inputs)]
    return httpx.Response(200, json={"data": data})


class Lanes(unittest.IsolatedAsyncioTestCase):
    async def test_query_does_not_wait_for_a_running_document_batch(self):
        release = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            if inputs[0].startswith("1"):  # the document batch: held until released
                await release.wait()
            return _vectors(inputs)

        document = DynamicBatcher(SERVER, "document", 256, 0, chunk_texts=8)
        query = DynamicBatcher(SERVER, "query", 256, 0)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            workers = [asyncio.create_task(b.run(client)) for b in (document, query)]
            try:
                docs = asyncio.create_task(document.submit(["1"] * 20, 2))
                await asyncio.sleep(0.01)
                result = await asyncio.wait_for(query.submit(["2"], 2), 1)
                self.assertEqual(len(result), 1)
                self.assertFalse(docs.done())
                release.set()
                self.assertEqual(len(await docs), 20)
            finally:
                for w in workers:
                    w.cancel()

    async def test_documents_are_sent_in_chunks_at_most_two_at_a_time(self):
        sizes, in_flight, peak = [], 0, 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            inputs = json.loads(request.content)["input"]
            sizes.append(len(inputs))
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _vectors(inputs)

        batcher = DynamicBatcher(SERVER, "document", 256, 0, chunk_texts=8)
        texts = [str(i) for i in range(1, 21)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            vectors = await batcher._embed_chunked(client, texts)
        self.assertEqual(sorted(sizes), [4, 8, 8])
        self.assertEqual(peak, 2)
        self.assertEqual(vectors[:, 0].tolist(), [float(t) for t in texts])

    async def test_a_failing_chunk_keeps_its_status(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            if "9" in inputs:
                return httpx.Response(503, json={"error": {"message": "loading"}})
            return _vectors(inputs)

        batcher = DynamicBatcher(SERVER, "document", 256, 0, chunk_texts=8)
        texts = [str(i) for i in range(1, 21)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ModelServerError) as cm:
                await batcher._embed_chunked(client, texts)
        self.assertEqual(cm.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
