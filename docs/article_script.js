// 获取所有的 h3 和 h4 元素
const sidebar = document.getElementById('dynamic-sidebar');
const elements = Array.from(document.querySelectorAll('h2, h3, h4')); // 获取所有 h2, h3 和 h4 元素

// 动态生成目录项
elements.forEach(element => {
    // 如果没有 id，则为该元素生成一个 id
    if (!element.id) {
        element.id = `${element.tagName.toLowerCase()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    const link = document.createElement('a');
    link.href = `#${element.id}`;
    link.textContent = element.textContent;

    const listItem = document.createElement('li');
    // 如果是 h4 元素，添加缩进
    if (element.tagName.toLowerCase() === 'h4') {
        listItem.style.marginLeft = '20px'; // 控制缩进
    }
    listItem.appendChild(link);

    sidebar.appendChild(listItem);
});

// 高亮当前目录项
function highlightCurrentLink(targetId) {
    const links = sidebar.querySelectorAll('a');
    links.forEach(link => {
        if (link.getAttribute('href') === `#${targetId}`) {
            link.classList.add('highlight-link');
        } else {
            link.classList.remove('highlight-link');
        }
    });
}

// 页面滚动时更新高亮
window.addEventListener('scroll', () => {
    let found = false;
    elements.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.top <= window.innerHeight / 2 && rect.bottom >= 0) {
            if (!found) {
                highlightCurrentLink(el.id);
                found = true;
            }
        }
    });
});

// 为每个目录链接添加点击事件
const links = sidebar.querySelectorAll('a');
links.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
            highlightCurrentLink(targetId);
        }
    });
});
