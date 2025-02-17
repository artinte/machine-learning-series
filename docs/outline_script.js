function toggleChildChapters(event, chapterId) {
    event.preventDefault();
    const chapter = document.getElementById(chapterId);
    if (chapter.style.display === "none" || !chapter.style.display) {
        chapter.style.display = "block";
    } else {
        chapter.style.display = "none";
    }
}

// 更新右侧导航，显示当前h3的小节
function updateRightSidebar(currentH3Id) {
    const sidebar = document.getElementById("dynamic-sidebar");
    sidebar.innerHTML = ""; // 清空当前内容

    // 查找当前h3下的所有h4元素
    const h3Element = document.getElementById(currentH3Id);
    let nextSibling = h3Element.nextElementSibling;
    const h4Elements = [];

    // 获取所有h4标签，直到遇到下一个h3标签
    while (nextSibling && nextSibling.tagName !== "H3") {
        if (nextSibling.tagName === "H4") {
            h4Elements.push(nextSibling);
        }
        nextSibling = nextSibling.nextElementSibling;
    }

    // 为每个h4元素创建导航项
    h4Elements.forEach(h4 => {
        const li = document.createElement("li");
        li.className = "nav-item";
        li.innerHTML = `<a class="nav-link" href="#${h4.id}" data-id="${h4.id}">${h4.innerText}</a>`;
        sidebar.appendChild(li);
    });

    // 默认高亮第一个小节
    if (h4Elements.length > 0) {
        highlightActiveSubtitle(h4Elements[0].id);
    }
}

// 高亮当前h4小节
function highlightActiveSubtitle(subtitleId) {
    const links = document.querySelectorAll("#dynamic-sidebar a");
    links.forEach(link => {
        if (link.getAttribute("data-id") === subtitleId) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });
}

// 监听h3小节滚动，显示对应的h4小节
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const currentH3Id = entry.target.id;
            updateRightSidebar(currentH3Id);
        }
    });
}, {
    threshold: 0.5 // 当h3元素有一半进入视口时触发
});

// 观察所有h3元素
const h3Elements = document.querySelectorAll("h3");
h3Elements.forEach(h3 => {
    observer.observe(h3);
});
