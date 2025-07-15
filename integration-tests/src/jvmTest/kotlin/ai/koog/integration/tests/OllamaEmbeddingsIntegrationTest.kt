package ai.koog.integration.tests

import ai.koog.embeddings.local.OllamaEmbeddingModels
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.extension.ExtendWith
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

@ExtendWith(OllamaTestFixtureExtension::class)
class OllamaEmbeddingsIntegrationTest {

    companion object {
        @field:InjectOllamaTestFixture
        private lateinit var fixture: OllamaTestFixture
        private val client get() = fixture.client
    }

    @Test
    fun testEmbed() = runTest {
        client.getModelOrNull(OllamaEmbeddingModels.NOMIC_EMBED_TEXT.id, true);

        val text = "This is a test text for embedding."
        val embedding = client.embed(text, OllamaEmbeddingModels.NOMIC_EMBED_TEXT)

        // Verify the embedding is not null and has the expected structure
        assertNotNull(embedding)
        assertTrue(embedding.isNotEmpty(), "Embedding should not be empty")

        // OpenAI embeddings typically have 1536 dimensions for text-embedding-ada-002
        // But we'll just check that it has a reasonable number of dimensions
        assertTrue(embedding.size > 100, "Embedding should have a reasonable number of dimensions")

        // Check that the embedding values are within a reasonable range
        embedding.forEach { value ->
            assertTrue(value.isFinite(), "Embedding values should be finite")
        }

        println("Embedding: $embedding")
    }

    @Test
    fun testEmbedWithDimensions() = runTest {
        client.getModelOrNull(OllamaEmbeddingModels.NOMIC_EMBED_TEXT.id, true);

        val text = "This is a test text for embedding."
        val embedding = client.embed(text, OllamaEmbeddingModels.NOMIC_EMBED_TEXT, 1024);

        // Verify the embedding is not null and has the expected structure
        assertNotNull(embedding)
        assertTrue(embedding.isNotEmpty(), "Embedding should not be empty")

        // Check that the embedding values are within a reasonable range
        embedding.forEach { value ->
            assertTrue(value.isFinite(), "Embedding values should be finite")
        }

        // Ollama doesn't support dimension configuration yet. The `dimensions` input will be ignored.
        // Verify embedding dimension not changed.
        assertEquals(embedding.size, 768)

        println("Embedding: $embedding")
    }
}
