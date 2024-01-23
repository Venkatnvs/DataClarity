const emailField=document.querySelector('#inputemail');
const emailfeedback=document.querySelector('.invalid-email-feedback');
const emailsuccess=document.querySelector('.email-success');
const submitbtn=document.querySelector('.submit-btn');


emailField.addEventListener("keyup", (e) => {
    const emailVal = e.target.value;

    emailsuccess.textContent=`Checking ${emailVal}`;

    emailField.classList.remove("is-invalid");
    emailfeedback.style.display='none';

    if (emailVal.length > 0) {
        emailsuccess.style.display='block';
        fetch("/utils/validate-email",{
            body:JSON.stringify({ email : emailVal}),method:"POST",
        }).then(res=>res.json()).then(data=>{
            console.log('data',data);
            emailsuccess.style.display='none';
            if (data.email_error){
                submitbtn.disabled = true;
                emailField.classList.add("is-invalid");
                emailfeedback.style.display='block';
                emailfeedback.innerHTML=`<p>${data.email_error}</p>`
            }else{
                submitbtn.removeAttribute('disabled');
            }
        });
    }

});