$(".toggle-password").click(function () {
    $(this).toggleClass("fa-eye fa-eye-slash");
    var input = $($(this).attr("toggle"));
    if (input.attr("type") == "password") {
      input.attr("type", "text");
    } else {
      input.attr("type", "password");
    }
  });
  window.onload = () =>{
    const allToolTips  = document.querySelectorAll(".nvs-tooltip");
      allToolTips.forEach(tt => {
          new bootstrap.Tooltip(tt)
    });
  }
  document.addEventListener('DOMContentLoaded', () => {
    "use strict";
    const preloader = document.querySelector('#preloader');
    if (preloader) {
      window.addEventListener('load', () => {
        preloader.remove();
      });
    }
    function aos_init() {
      AOS.init({
        duration: 1000,
        easing: 'ease-in-out',
        once: true,
        mirror: false
      });
      AOS.init({disable: 'mobile'});
    }
    window.addEventListener('load', () => {
      aos_init();
    }); 
  });