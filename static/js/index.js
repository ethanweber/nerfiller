window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  // for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
  //   var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
  //   interp_images[i] = new Image();
  //   interp_images[i].src = path;
  // }
}

function setInterpolationImage(i) {
  // var image = interp_images[i];
  // image.ondragstart = function() { return false; };
  // image.oncontextmenu = function() { return false; };
  // $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: false,
			autoplay: false,
			autoplaySpeed: 3000,
    }

    var options1 = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: false,
			autoplay: false,
			autoplaySpeed: 3000,
    }

    var options2 = {
			slidesToScroll: 1,
			slidesToShow: 2,
			loop: true,
			infinite: false,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
    var carousels = bulmaCarousel.attach('.carousel1', options1);
    var carousels = bulmaCarousel.attach('.carousel2', options2);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

var dataset = 'dumptruck';
var method = 'masked-nerf';
var activeVidID = 0;

function setComparisonVideo() {
  // swap video to avoid flickering
  activeVidID = 1 - activeVidID;
  var video_active = document.getElementById("comparison-video-" + activeVidID);
  video_active.src = "static/videos_comparison/" + dataset + "-" + method + ".mp4";
  video_active.load();

  // for all the buttons, set is-dark
  var buttons = document.getElementsByClassName("button");
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].classList.remove("is-light");
    buttons[i].classList.add("is-dark");
  }
  // for the selected buttons, set is-light
  var button = document.getElementById(dataset);
  button.classList.remove("is-dark");
  button.classList.add("is-light");
  var button = document.getElementById(method);
  button.classList.remove("is-dark");
  button.classList.add("is-light");
}

function setDataset(datasetName) {
  dataset = datasetName;
  setComparisonVideo();
}

function setMethod(methodName) {
  method = methodName;
  setComparisonVideo();
}