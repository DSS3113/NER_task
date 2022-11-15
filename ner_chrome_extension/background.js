
chrome.runtime.onMessage.addListener((msg, sender, response) => {
    if(msg.name == "getNamedEntities") {
        fetch('http://127.0.0.1:5000/', {
            method: "POST",
            headers: {
                'Content-type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({"body_text": msg.body_text, "body_html": msg.body_html})
        }).then(r => r.json().then(data => { response(data); }));   
    }
    return true;
});