document.addEventListener('DOMContentLoaded', (event) => {
    var links = document.getElementsByClassName('headerbtn');
    for (var i = 0; i < links.length; ++i) {
        links[i].setAttribute('target', '_blank');
    }
  })