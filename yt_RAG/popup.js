const sendbutton=document.getElementById("send")
const questionin=document.getElementById("question")
const responseDiv = document.getElementById("response")

sendbutton.addEventListener("click",()=>{
    const questext=questionin.value.trim()

    if(questext === "") {
        console.log("No question asked :)")
        return
    }
    chrome.tabs.query({active:true,currentWindow:true},function(tabs) {
        if (tabs && tabs[0]) {
            const currenturl=tabs[0].url
            const vidid=extract_video_id(currenturl)
            const payload = {
                url: currenturl,
                video_id: vidid,
                question: questext
            }

            console.log(payload)

            fetch("http://localhost:8000/ask",{
                method:"POST",
                headers:{
                    "Content-Type":"application/json"
                },
                body:JSON.stringify(payload)
            })
            .then(Response=>{
                if(!Response.ok) {
                    throw new Error("Backend error: " + Response.status)
                }
                return Response.json()
            })
            .then(data=>{
                console.log("backend response recived",data)
                responseDiv.innerText = data.answer;
            })
            .catch(error=>{
                console.error("error occured ",error)
                responseDiv.innerText = "Error could not reach backend";
            })
        }
    })
    questionin.value=""
});

function extract_video_id(url) {
    if(!url) return null

    try{
        const parsedurl=new URL(url)

        if(parsedurl.hostname.includes("youtube.com")) {
            return parsedurl.searchParams.get("v")
        }
        if (parsedurl.hostname === "youtu.be") {
      
            return parsedurl.pathname.slice(1);
        }
    } catch (error) {
        console.log(error)
    }
    return null
}