import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { config } from "dotenv";
import readline from "readline";
import { promisify } from "util";
//openai
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

config();

const pinecone = new Pinecone();

const indexName = "test";

async function loadPdfandSplit(path) {
  const pdfPath = path;

  const loader = new PDFLoader(pdfPath);
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 10,
  });

  const chunks = await splitter.createDocuments(
    docs.map((doc) => doc.pageContent)
  );
  return chunks;
}

async function storeInVectorStore(chunks, question) {
  const embeddings = new OpenAIEmbeddings();

  const index = pinecone.Index(indexName);
  const vectorstore = await PineconeStore.fromDocuments(chunks, embeddings, {
    pineconeIndex: index,
  });
  const searches = await vectorstore.similaritySearch(question);

  return searches;
}

async function generatePrompt(role, searches, question) {
  let context = "";
  searches.forEach((search) => {
    context = context + "\n\n" + search.pageContent;
  });

  const prompt = ChatPromptTemplate.fromTemplate(`
    you are a {role}, please answer within your role and context, else apologize
    Role:{role}
    Tone:"pleasant and polite"
    Context: {context}
    Question: {question}`);

  const formattedPrompt = await prompt.format({
    role: role,
    context: context,
    question: question,
  });
  return formattedPrompt;
}

async function generateResult(prompt) {
  const model = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0.2,
    maxTokens: 500,
  });

  const response = await model.invoke(prompt);
  return response;
}

async function run() {
  const splittedData = await loadPdfandSplit("TheRulesofLife.pdf");
  const reader = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const questionAsync = promisify(reader.question).bind(reader);
  let role = "book expert";
  while (true) {
    const question = await questionAsync("user: ");
    if (question.toLowerCase() == "exit") {
      console.log("Goodbye!");
      reader.close();
      process.exit();
    } else {
      const searches = await storeInVectorStore(splittedData, question);
      const prompt = await generatePrompt(role, searches, question);
      const result = await generateResult(prompt);
      console.log("\nAI: ", result.content);
    }
  }
}
run();
