//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag"); //得到ID为file-drag的对象
var fileSelect = document.getElementById("file-upload");

// Add event listeners 添加事件侦听器
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour 防止默认行为
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting 句柄文件选择
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use 用于函数的网页元素
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var method1 = document.getElementById("method1");
var method2 = document.getElementById("method2");
var data_all;
var data_detail;
//public static final String data;
//var complex = document.getElementById("complex");


//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // action for the submit button 提交按钮的操作
  console.log("submit");//在浏览器中打印对象
  if (!imagePreview.src ) {
    window.alert("Please select an image before submit.");
    return;
  }

  //loader.classList.remove("hidden");  // 隐藏的识别图片打开
  //imageDisplay.classList.add("loading");
  var selectmethod1 = method1.checked;
  var selectmethod2 = method2.checked;

//  if(selecteasy || selectcomplex){
//    window.alert("Please select a picture type before submit.");
//    return;
//  }
  if (selectmethod1){
  // call the predict function of the backend 调用后端的预测函数
  Image_super_resolution(imagePreview.src);}
  else if (selectmethod2){
  Image_super_resolution(imagePreview.src);}
  console.log("data");//在浏览器中打印对象
  }


function clearImage() {  // 点击清除按钮
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  //imageDisplay.src = "";
  //predResult.innerHTML = "";
  //document.getElementById("text").innerHTML= "";
  hide(imagePreview);
  //hide(imageDisplay);
  //hide(loader);
  //hide(predResult);
  show(uploadCaption);


  //imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  // show the preview of the image 显示图像预览
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    //predResult.innerHTML = "";
    //imageDisplay.classList.remove("loading");

    //displayImage(reader.result, "image-display");
  };
}

function displayImage(image, id) {
  // display image on given id <img> element 在给定id＜img＞元素上显示图像
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  hide(loader);
  predResult.innerHTML = data.result;
//  predResult.innerHTML = data.probability;
  show(predResult);
}
//隐藏元素
function hide(el) {
  // hide an element
  el.classList.add("hidden");
}
//显示元素
function show(el) {
  // show an element
  el.classList.remove("hidden");
}


///选择超分方法
function Image_super_resolution(image) {
  fetch("/predict_easy", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)  //将对象转为JSON字符串
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
//          data_all = data;
//          data_detail = data.ten;
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
//  return data;
}
function Image_super_resolution(fileInput) {
  // 创建一个新的 FormData 对象
  var formData = new FormData();
  // 添加文件到 FormData 对象
  formData.append('file', fileInput.files[0]);

  // 使用 fetch API 发送 POST 请求
  fetch("/predict_easy", {
    method: "POST",
    body: formData  // 使用 FormData 对象作为请求体
  })
  .then(response => {
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    return response.json();  // 解析 JSON 响应
  })
  .then(data => {
    // 处理响应数据，例如显示处理后的图片
    console.log(data); // 这里应该根据你的后端响应格式来处理
    // 假设后端返回了处理后的图片 URL
    document.getElementById('image-preview').src = data.image_url;
  })
  .catch(error => {
    // 处理错误情况
    console.error('There has been a problem with your fetch operation:', error);
    alert('上传图片失败，请重试。');
  });
}