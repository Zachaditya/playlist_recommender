document.addEventListener("DOMContentLoaded", () => {
  fetch("/songs")
    .then((response) => response.json())
    .then((songs) => {
      const container = document.getElementById("songsContainer");

      songs.forEach((song) => {
        const frame = document.createElement("div");
        frame.classList.add("frame");

        frame.innerHTML = `
                    <img src="${song.img}" alt="${song.name}">
                    <p>${song.name}</p>
                    <button class="btn btn-success" onclick="addToCart('${song.name}')">Add to Playlist</button>

                `;

        container.appendChild(frame);
      });

      updateArrows();
      container.addEventListener("scroll", updateArrows);
    })
    .catch((error) => console.error("Error fetching songs:", error));
  loadCart();
});

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("cartForm");
  const hiddenInput = document.getElementById("cartData");

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    const cart = JSON.parse(localStorage.getItem("cart")) || [];
    const songNames = cart.map((song) => song.name);

    hiddenInput.value = JSON.stringify(songNames);

    form.submit();
  });
});

function scrollCarousel(direction) {
  const container = document.getElementById("songsContainer");
  const scrollAmount = 1100;

  container.scrollBy({ left: direction * scrollAmount, behavior: "auto" });

  setTimeout(updateArrows, 500);
}

function updateArrows() {
  const container = document.getElementById("songsContainer");
  const leftArrow = document.querySelector(".arrow-left");
  const rightArrow = document.querySelector(".arrow-right");

  leftArrow.style.opacity = container.scrollLeft > 0 ? "1" : "0.5";
  rightArrow.style.opacity =
    container.scrollLeft + container.clientWidth < container.scrollWidth
      ? "1"
      : "0.5";
}

function addToCart(name, artist) {
  let cart = JSON.parse(localStorage.getItem("cart")) || [];

  if (!cart.some((song) => song.name === name)) {
    cart.push({ name, artist });
    localStorage.setItem("cart", JSON.stringify(cart));
    alert("Successfully added to Playlist");

    // Also send to server
    fetch("/add-to-cart", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name: name, artist: artist }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Server responded:", data);
      })
      .catch((error) => console.error("Error sending to server:", error));
  } else {
    alert("This song is already in your cart!");
  }
  loadCart();
}

function loadCart() {
  let cart = JSON.parse(localStorage.getItem("cart")) || [];
  const cartList = document.getElementById("cartItems");

  cartList.innerHTML = "";
  cart.forEach((song, index) => {
    let listItem = document.createElement("li");
    listItem.classList.add(
      "list-group-item",
      "d-flex",
      "justify-content-between",
      "align-items-center"
    );
    listItem.innerHTML = `
                                        
              
                ${song.name} - ${song.artist}
                <button class="btn btn-danger btn-sm" onclick="removeFromCart(${index})" style= 'height: 50%; width: 50%'>Remove</button>
            `;
    cartList.appendChild(listItem);
  });
}

function removeFromCart(index) {
  let cart = JSON.parse(localStorage.getItem("cart")) || [];
  cart.splice(index, 1);
  localStorage.setItem("cart", JSON.stringify(cart));
  alert("Successfully removed song from your cart!");
  loadCart();
}

fetch("/artists")
  .then((response) => response.json())
  .then((artists) => {
    const container = document.getElementById("artistsList");

    artists.forEach((artist) => {
      const card = document.createElement("div");
      card.classList.add("col-md-4", "mb-4");
      card.innerHTML = `
        <div class = "card_section">
            <div class =  "card">
                <img src="${artist["img"]}" class="card-img-top song-image" alt="${artist.name}">
                <p class= "card p">${artist.artist}</p>  
            </div>
        </div>
            `;
      container.appendChild(card);
    });
  })
  .catch((error) => console.error("Error fetching artists:", error));
