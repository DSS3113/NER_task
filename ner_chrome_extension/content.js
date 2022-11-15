let body_tag = document.getElementsByTagName("body")[0];
let body_html = body_tag.innerHTML;
let body_text = body_tag.innerText;


// highlightNamedEntities()


chrome.runtime.sendMessage({
    name: "getNamedEntities", 
    body_text: body_text,
    body_html: body_html
}, (res) => {
    if(res['body_html']) {
        body_tag.innerHTML = res["body_html"]
    }
});
