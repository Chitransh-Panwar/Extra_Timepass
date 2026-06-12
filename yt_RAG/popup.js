const sendbutton = document.getElementById("send")
const questionin = document.getElementById("question")
const chatHistory = document.getElementById("chat-history")
const clearBtn = document.getElementById("clear-btn")
let currentSessionId = null

function addMessage(text, role) {
  const msg = document.createElement("div")
  msg.className = `msg ${role}`
  msg.innerText = text
  chatHistory.appendChild(msg)
  chatHistory.scrollTop = chatHistory.scrollHeight
  return msg
}

function clearChat() {
  chatHistory.innerHTML = ""
  addMessage("Ask me anything about this page or video!", "assistant")
  currentSessionId = null
}

clearBtn.addEventListener("click", clearChat)

sendbutton.addEventListener("click", () => {
  const questext = questionin.value.trim()
  if (questext === "") return

  addMessage(questext, "user")
  const thinkingMsg = addMessage("Thinking...", "thinking")
  sendbutton.disabled = true
  questionin.value = ""

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs && tabs[0]) {
      const currenturl = tabs[0].url
      const vidid = extract_video_id(currenturl)
      const payload = {
        url: currenturl,
        video_id: vidid,
        question: questext,
        session_id: currentSessionId
      }

      fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
      .then(response => {
        if (!response.ok) throw new Error("Backend error: " + response.status)
        return response.json()
      })
      .then(data => {
        thinkingMsg.remove()
        addMessage(data.answer, "assistant")
        if (data.session_id) currentSessionId = data.session_id
        sendbutton.disabled = false
      })
      .catch(error => {
        thinkingMsg.remove()
        addMessage("Error: could not reach backend.", "assistant")
        console.error(error)
        sendbutton.disabled = false
      })
    } else {
      sendbutton.disabled = false
    }
  })
})

function extract_video_id(url) {
  if (!url) return null
  try {
    const parsedurl = new URL(url)
    if (parsedurl.hostname.includes("youtube.com")) return parsedurl.searchParams.get("v")
    if (parsedurl.hostname === "youtu.be") return parsedurl.pathname.slice(1)
  } catch (e) { console.log(e) }
  return null
}