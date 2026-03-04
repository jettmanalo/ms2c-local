import React from "react";

export const SuccessAlertLightFill = () => (<div
    className="flex items-center justify-between text-blue-600 max-w-80 w-full bg-blue-600/10 h-10 shadow">
    <div className="h-full w-1.5 bg-blue-600"></div>
    <div className="flex items-center">
        <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="icon line">
            <path style="fill:none;stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;stroke-width:1.95"
                  d="M11.95 16.5h.1"/>
            <path d="M3 12a9 9 0 0 1 9-9h0a9 9 0 0 1 9 9h0a9 9 0 0 1-9 9h0a9 9 0 0 1-9-9m9 0V7"
                  style="fill:none;stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;stroke-width:1.5"/>
        </svg>
        <p className="text-sm mb-2">Success! Your task is fully completed.</p>
    </div>
    <button type="button" aria-label="close" className="active:scale-90 transition-all mr-3">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 5L5 15M5 5L15 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"
                  strokeLinejoin="round"/>
        </svg>
    </button>
</div>)

export const Notification = () => (<div
    className="bg-white divide-gray-300/60 flex divide-x pl-3 text-sm rounded border border-gray-300/60">
    <div className="py-2.5 flex justify-center flex-col pr-30">
        <h3 className="text-gray-700 font-medium">Receive notifications</h3>
        <p className="text-gray-500 max-w-64">Notifications may include alerts, sounds, and badges.</p>
    </div>
    <div className="flex flex-col items-center divide-y divide-gray-500/30">
        <button type="button"
                className="text-indigo-500 font-medium cursor-pointer h-full w-28 hover:bg-indigo-500/10 transition-all">
            Reply
        </button>
        <button type="button"
                className="text-gray-500 font-medium cursor-pointer h-full w-28 hover:bg-gray-300/10 transition-all">
            Don't allow
        </button>
    </div>
</div>)

export const HiringBadge = () => (<div
    className="flex items-center gap-2 border border-indigo-200 rounded-full p-1 pr-3 text-sm text-slate-500">
    <span className="text-indigo-600 pt-2 font-medium pr-1">We're hiring</span>
    <div className="h-6 w-px bg-gray-300"></div>
    <a href="#" className="flex items-center gap-1 px-1">
        See open positions
        <svg className="mt-1" width="6" height="9" viewBox="0 0 6 9" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="m1 1 4 3.5L1 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
                  strokeLinejoin="round"/>
        </svg>
    </a>
</div>)

export const Breadcrumb = () => (<div
    className="flex flex-wrap items-center justify-center space-x-2 text-sm text-gray-500 font-medium bg-white py-14 px-4 border border-gray-200 rounded-lg">
    <button type="button" aria-label="Home">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M16 7.609c.352 0 .69.122.96.343l.111.1 6.25 6.25v.001a1.5 1.5 0 0 1 .445 1.071v7.5a.89.89 0 0 1-.891.891H9.125a.89.89 0 0 1-.89-.89v-7.5l.006-.149a1.5 1.5 0 0 1 .337-.813l.1-.11 6.25-6.25c.285-.285.67-.444 1.072-.444Zm5.984 7.876L16 9.5l-5.984 5.985v6.499h11.968z"
                fill="#898d95ff" stroke="#898d95ff" strokeWidth=".094"/>
        </svg>
    </button>
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
            d="m14.413 10.663-6.25 6.25a.939.939 0 1 1-1.328-1.328L12.42 10 6.836 4.413a.939.939 0 1 1 1.328-1.328l6.25 6.25a.94.94 0 0 1-.001 1.328"
            fill="#CBD5E1"/>
    </svg>
    <a href="#">Link Item</a>
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
            d="m14.413 10.663-6.25 6.25a.939.939 0 1 1-1.328-1.328L12.42 10 6.836 4.413a.939.939 0 1 1 1.328-1.328l6.25 6.25a.94.94 0 0 1-.001 1.328"
            fill="#CBD5E1"/>
    </svg>
    <a href="#" className="text-indigo-500">Link Item</a>
</div>)

export const DownloadButton1 = () => (<button type="button"
                                              className="flex items-center justify-between text-gray-800/80 text-sm h-10 w-36 py-4 bg-white border active:scale-95 transition border-gray-500/30">
    Download
    <div className="bg-gray-500/20 h-full flex items-center justify-center px-3">
        <svg width="20" height="21" viewBox="0 0 20 21" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M17.5 13.125v3.5c0 .464-.176.91-.488 1.237a1.63 1.63 0 0 1-1.179.513H4.167c-.442 0-.866-.184-1.179-.513a1.8 1.8 0 0 1-.488-1.237v-3.5M5.833 8.75 10 13.125m0 0 4.167-4.375M10 13.125v-10.5"
                stroke="#1F2937" strokeOpacity=".8" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
    </div>
</button>)

export const SimpleCardWithButton = () => (<div
    className="p-4 bg-white border border-gray-200 hover:-translate-y-1 transition duration-300 rounded-lg shadow shadow-black/10 max-w-80">
    <img className="rounded-md max-h-40 w-full object-cover"
         src="https://images.unsplash.com/photo-1560264418-c4445382edbc?q=80&w=400" alt="officeImage"/>
    <p className="text-gray-900 text-xl font-semibold ml-2 mt-4">
        Your Card Title
    </p>
    <p className="text-zinc-400 text-sm/6 mt-2 ml-2 ml-2">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et
        dolore..
    </p>
    <button type="button"
            className="bg-indigo-600 hover:bg-indigo-700 transition cursor-pointer mt-4 mb-3 ml-2 px-6 py-2 font-medium rounded-md text-white text-sm">
        Learn More
    </button>
</div>)

export const UserProfileCardRounded2 = () => (<div
    className="bg-white rounded-2xl pb-4 overflow-hidden border border-gray-200 hover:-translate-y-1 transition duration-300">
    <div className="w-64 flex justify-center pt-1">
        <div className="w-28 h-28 rounded-full overflow-hidden">
            <img className="h-32 object-cover object-top"
                 src="https://images.unsplash.com/photo-1633332755192-727a05c4013d?q=80&w=200" alt="userImage2"/>
        </div>
    </div>
    <div className="flex flex-col items-center">
        <p className="font-medium mt-3">Kelvin John</p>
        <p className="text-gray-500 text-sm">kelvin.john@gmail.com</p>
        <button
            className="border text-sm text-gray-500 border-gray-200 hover:bg-gray-100 transition cursor-pointer px-6 py-1 rounded-full mt-5 flex items-center justify-center gap-1">
            <svg className="mt-0.5" width="13" height="13" viewBox="0 0 13 13" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
                <path
                    d="m7.107 11.684.31-.521-.736-.436-.309.522zm-2.28-.521.308.521.735-.435-.309-.522zm1.545.086a.297.297 0 0 1-.502 0l-.735.435a1.15 1.15 0 0 0 1.972 0zM5.267.854h1.708V0H5.267zm6.121 4.413v.57h.854v-.57zm-10.534.57v-.57H0v.57zm-.854 0c0 .657 0 1.171.028 1.586.029.42.088.768.221 1.09l.79-.327c-.084-.2-.133-.446-.159-.82-.026-.38-.026-.86-.026-1.53zm3.731 3.838c-.715-.012-1.09-.058-1.383-.18l-.327.79c.459.19.98.232 1.695.244zM.249 8.513c.333.802.97 1.44 1.772 1.772l.327-.79a2.42 2.42 0 0 1-1.31-1.309zm11.14-2.677c0 .67-.001 1.15-.027 1.53-.026.374-.075.62-.158.82l.79.327c.133-.322.192-.67.22-1.09.028-.415.028-.93.028-1.587zM8.525 10.53c.715-.012 1.237-.054 1.695-.244l-.327-.79c-.293.122-.668.168-1.383.18zm2.678-2.343a2.42 2.42 0 0 1-1.31 1.31l.327.789a3.27 3.27 0 0 0 1.772-1.772zM6.975.854c.94 0 1.616 0 2.142.05.52.05.852.145 1.116.307l.446-.729C10.259.225 9.78.11 9.199.054 8.621 0 7.898 0 6.974 0zm5.267 4.413c0-.924 0-1.646-.054-2.223-.056-.583-.17-1.06-.428-1.48l-.728.446c.161.264.256.595.306 1.115.05.527.05 1.202.05 2.142zm-2.01-4.056c.326.2.6.473.8.799l.728-.447c-.27-.44-.64-.81-1.081-1.08zM5.268 0c-.924 0-1.646 0-2.223.054-.583.056-1.06.17-1.48.428l.446.729c.264-.162.595-.257 1.115-.306.527-.05 1.202-.05 2.142-.05zM.854 5.267c0-.94 0-1.615.05-2.142.05-.52.145-.851.307-1.115l-.729-.447c-.257.421-.372.898-.428 1.481C0 3.621 0 4.344 0 5.267zm.71-4.785A3.3 3.3 0 0 0 .482 1.563l.729.447c.2-.326.473-.6.799-.8zM5.56 10.728a6 6 0 0 0-.316-.503 1.3 1.3 0 0 0-.388-.368l-.43.739a.4.4 0 0 1 .128.131c.07.095.147.226.271.436zm-1.845-.199c.25.004.409.008.53.02a.45.45 0 0 1 .182.047l.429-.739a1.3 1.3 0 0 0-.518-.156c-.169-.019-.374-.022-.608-.026zm3.7.634c.124-.21.202-.34.271-.436a.4.4 0 0 1 .128-.131l-.43-.739a1.3 1.3 0 0 0-.388.368c-.099.135-.2.307-.316.502zM8.51 9.675c-.234.004-.439.007-.608.026-.178.02-.351.06-.518.156l.43.739a.45.45 0 0 1 .182-.046 6 6 0 0 1 .529-.02z"
                    fill="#6B7280"/>
                <path d="M3.844 5.552h.005m2.268 0h.005m2.272 0H8.4" stroke="#6B7280" strokeWidth="1.5"
                      strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <p className="mb-1">message</p>
        </button>
    </div>
</div>)

export const ReceiptCard2 = () => (<div
    className="bg-indigo-500/5 border border-gray-500/20 text-sm text-gray-500 flex flex-col items-center w-80 rounded-lg">
    <div className="flex items-center justify-between w-full px-4 py-2">
        <div className="flex items-center justify-between gap-3">
            <div className="bg-white p-1.5 rounded border border-gray-500/30">
                <img className="h-9"
                     src="https://raw.githubusercontent.com/prebuiltui/prebuiltui/main/assets/dummyLogo/dummyFavicon.svg"
                     alt="dummyFavicon"/>
            </div>
            <p className="text-lg text-gray-800">Router</p>
        </div>
        <button type="button" aria-label="more">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                    d="M11 11.917a.917.917 0 1 0 0-1.833.917.917 0 0 0 0 1.833M11 5.5a.917.917 0 1 0 0-1.833.917.917 0 0 0 0 1.833m0 12.834a.917.917 0 1 0 0-1.834.917.917 0 0 0 0 1.834"
                    stroke="#6B7280" strokeOpacity=".8" strokeWidth="1.5" strokeLinecap="round"
                    strokeLinejoin="round"/>
            </svg>
        </button>
    </div>
    <div
        className="flex flex-col items-center gap-2 w-full p-4 pb-2 rounded-b-lg bg-white border-t border-gray-500/20">
        <div className="flex items-center w-full justify-between">
            <p>Last invoice</p>
            <p>December 24, 2023</p>
        </div>
        <div className="w-full h-px bg-gray-300/60"></div>
        <div className="flex items-center w-full justify-between">
            <p>Amount</p>
            <div className="flex items-center gap-2">
                <p>$800.00</p>
                <p className="bg-red-500/20 px-21 py-0.5 rounded border border-red-500/30 text-red-600">Overdue</p>
            </div>
        </div>
    </div>
</div>)

export const PrivacyChoicesPanel = () => (<div
    className="w-80 bg-white text-gray-500 p-4 md:p-6 rounded-lg border border-gray-500/30 text-sm">
    <h2 className="text-gray-800 text-xl font-medium pb-3">We care about your privacy</h2>
    <p>This website uses cookies for functionality, analytics, and marketing. By accepting, you agree to our <a
        href="#" className="font-medium underline">Cookie Policy</a>.</p>
    <div className="flex items-center justify-center mt-6 gap-4">
        <button type="button"
                className="font-medium px-8 border border-gray-500/30 py-2 rounded hover:bg-blue-500/10 transition active:scale-95">Decline
        </button>
        <button type="button"
                className="font-medium pt-8 border border-gray-500/30 py-2 rounded hover:bg-blue-500/10 transition active:scale-95">Settings
        </button>
    </div>
    <button type="button"
            className="bg-indigo-600 w-full py-2 mt-4 rounded text-white font-medium active:scale-95 transition">Accept
    </button>
</div>)

export const TeamMemberCard3 = () => (<div
    className="text-sm text-gray-500 w-80 divide-y divide-gray-500/30 border border-gray-500/30 rounded bg-white">
    <div className="flex items-start justify-between p-3">
        <div>
            <div className="flex items-center space-x-2">
                <h2 className="text-lg text-gray-800">Donald Jackman</h2>
                <p className="bg-green-500/20 px-2 py-0.5 rounded-full text-xs text-green-600 border border-green-500/30">Admin</p>
            </div>
            <p>Content Creator</p>
        </div>
        <img className="h-10 w-10 rounded-full"
             src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?q=80&w=100&h=100&auto=format&fit=crop"
             alt="userImage3"/>
    </div>
    <div className="flex items-center divide-x divide-gray-500/30">
        <button type="button" className="flex items-center justify-center gap-2 w-1/2 px-3">
            <svg width="18" height="14" viewBox="0 0 18 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                    d="M16.5 2.5c0-.825-.675-1.5-1.5-1.5H3c-.825 0-1.5.675-1.5 1.5m15 0v9c0 .825-.675 1.5-1.5 1.5H3c-.825 0-1.5-.675-1.5-1.5v-9m15 0L9 7.75 1.5 2.5"
                    stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Send Email
        </button>
        <button type="button" className="flex items-center justify-center gap-2 w-1/2 py-3">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                    d="M11.29 3.75a3.75 3.75 0 0 1 2.962 2.963M11.289.75a6.75 6.75 0 0 1 5.963 5.955m-.75 5.985v2.25a1.5 1.5 0 0 1-1.635 1.5 14.84 14.84 0 0 1-6.472-2.303 14.6 14.6 0 0 1-4.5-4.5 14.84 14.84 0 0 1-2.303-6.502A1.5 1.5 0 0 1 3.085 1.5h2.25a1.5 1.5 0 0 1 1.5 1.29 9.6 9.6 0 0 0 .525 2.108 1.5 1.5 0 0 1-.338 1.582l-.952.952a12 12 0 0 0 4.5 4.5l.952-.952a1.5 1.5 0 0 1 1.582-.338c.681.254 1.388.43 2.108.526a1.5 1.5 0 0 1 1.29 1.522"
                    stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Call Now
        </button>
    </div>
</div>)

export const CreateNewProjectForm = () => (<form
    className="bg-white text-gray-500 max-w-[340px] mx-4 p-6 text-left text-sm rounded-lg border border-gray-300/60">
    <label className="font-medium" htmlFor="email">Project Title</label>
    <input id="email" className="w-full border mt-1.5 mb-32 border-gray-500/30 outline-none rounded py-2.5 px-3"
           type="email" placeholder="Enter title" required/>
    <label className="font-medium" htmlFor="content">Content</label>
    <textarea rows="3" id="content"
              className="w-full resize-none border mt-1.5 border-gray-500/30 outline-none rounded py-2.5 px-3"
              type="email" placeholder="Enter content" required></textarea>
    <div className="flex items-center justify-between">
        <button type="submit" className="my-3 bg-indigo-500 py-2 px-5 rounded text-white font-medium">Post</button>
        <div className="space-x-0.5">
            <button type="button" aria-label="add">
                <svg width="24" height="24" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                        d="M10 8H8m0 0H6m2 0V6m0 2v2m3.333 4H4.667A2.667 2.667 0 0 1 2 11.333V4.667A2.667 2.667 0 0 1 4.667 2h6.666A2.667 2.667 0 0 1 14 4.667v6.666A2.667 2.667 0 0 1 11.333 14Z"
                        stroke="currentColor" strokeOpacity=".8" strokeLinecap="round"/>
                </svg>
            </button>
            <button type="button" aria-label="addPicture">
                <svg width="24" height="24" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                        d="M9.51 10.625 8.365 9.49c-.526-.522-.79-.783-1.092-.88a1.33 1.33 0 0 0-.82 0c-.303.097-.566.358-1.092.88l-2.666 2.685m6.815-1.55.228-.226c.537-.532.806-.799 1.114-.896.27-.086.562-.082.831.01.306.104.569.376 1.094.921l.557.566m-3.824-.375 2.637 2.684m-9.452-1.134c.02.173.056.31.117.43.128.251.332.455.583.583.285.145.659.145 1.405.145h6.4c.415 0 .714 0 .947-.024m-9.452-1.134c-.028-.237-.028-.544-.028-.975V4.8c0-.747 0-1.12.145-1.405.128-.251.332-.455.583-.583.285-.145.659-.145 1.405-.145h2.534m4.813 10.642c.187-.02.332-.056.459-.121.25-.128.454-.332.582-.583.146-.285.146-.658.146-1.405V8.667M12 6V4m0 0V2m0 2h2m-2 0h-2"
                        stroke="currentColor" strokeOpacity=".8" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </button>
            <button type="button" aria-label="notes">
                <svg width="24" height="24" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                        d="M2 6h12m-2.667 2.668-6.666-.001m2.222 2.667H4.667m0-9.334v1.333M11.333 2v1.333M4.133 14h7.734c.746 0 1.12 0 1.405-.145a1.34 1.34 0 0 0 .583-.583c.145-.285.145-.659.145-1.405v-6.4c0-.747 0-1.12-.145-1.406a1.33 1.33 0 0 0-.583-.582c-.285-.146-.659-.146-1.405-.146H4.133c-.746 0-1.12 0-1.405.146-.25.127-.455.331-.583.582C2 4.347 2 4.72 2 5.467v6.4c0 .746 0 1.12.145 1.405.128.25.332.455.583.583.285.145.659.145 1.405.145"
                        stroke="currentColor" strokeOpacity=".8" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </button>
        </div>
    </div>
</form>)

export const ForgotPasswordFromDark = () => (<div
    className="bg-white text-gray-500 max-w-96 mx-4 md:p-6 p-4 text-left text-sm rounded shadow-[0px_0px_10px_0px] shadow-black/10">
    <h2 className="text-2xl font-semibold mb-6 text-center text-gray-800">Forget Password?</h2>
    <label htmlFor="email">Email</label>
    <input id="email"
           className="w-full border mt-1 border-gray-500/30 focus:border-indigo-500 outline-none rounded py-2.5 px-4"
           type="email" placeholder="Enter your email"/>
    <button type="button"
            className="w-full my-21 bg-gray-800 active:scale-95 transition py-2.5 rounded text-white">Send Email
    </button>
    <p className="text-center mt-4">Don’t have an account? <span
        className="text-blue-500 underline">Signup Now</span></p>
</div>)

export const OtpVerificationForm = () => (<div
    className="flex flex-col items-center md:max-w-[423px] w-[380px] bg-white rounded-2xl shadow-lg p-6 sm:p-1">
    <p className="text-2xl font-semibold text-gray-900">Email Verify OTP</p>
    <p className="mt-2 text-sm text-gray-900/90 text-center">Enter the 6-digit code sent to your email ID.</p>

    <div className="grid grid-cols-6 gap-2 sm:gap-3 w-11/12 mt-8">
        <input type="text" maxLength="1"
               className="w-full h-12 bg-indigo-50 text-gray-900 text-xl rounded-md outline-none text-center"/>
        <input type="text" maxLength="1"
               className="w-full h-12 bg-indigo-50 text-gray-900 text-xl rounded-md outline-none text-center"/>
        <input type="text" maxLength="1"
               className="w-full h-12 bg-indigo-50 text-gray-900 text-xl rounded-md outline-none text-center"/>
        <input type="text" maxLength="1"
               className="w-full h-12 bg-indigo-50 text-gray-900 text-xl rounded-md outline-none text-center"/>
        <input type="text" maxLength="1"
               className="w-full h-12 bg-indigo-50 text-gray-900 text-xl rounded-md outline-none text-center"/>
        <input type="text" maxLength="1"
               className="w-full h-12 bg-indigo-50 text-gray-900 text-xl rounded-md outline-none text-center"/>
    </div>

    <button type="button"
            className="mt-8 w-full max-w-80 h-11 rounded-full text-white text-sm bg-indigo-500 hover:opacity-90 transition-opacity">
        Verify Email
    </button>
</div>)

export const UploadAreaWithTitle = () => (<div
    className="max-w-md w-full p-6 bg-white rounded-lg border border-gray-500/30 shadow-[0px_1px_15px_0px] shadow-black/10 text-sm">
    <div className="flex items-center justify-center w-11 h-11 bg-gray-500/10 rounded-full">
        <svg width="19" height="19" viewBox="0 0 19 19" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M7.124 11.083h4.75m5.541 3.959a1.584 1.584 0 0 1-1.583 1.583H3.165a1.583 1.583 0 0 1-1.583-1.583V3.958a1.583 1.583 0 0 1 1.583-1.583h3.959L8.707 4.75h7.125a1.583 1.583 0 0 1 1.583 1.583z"
                stroke="#2563EB" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
    </div>
    <h2 className="text-2xl text-gray-800 font-medium mt-3">Upload a file</h2>
    <p className="text-gray-500/80 mt-1">Attach the file below</p>
    <label htmlFor="fileInput"
           className="border-2 border-dotted border-gray-400 p-8 mt-6 flex flex-col items-center gap-4 cursor-pointer hover:border-blue-500 transition">
        <svg width="31" height="31" viewBox="0 0 31 31" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M18.085 2.583H7.75a2.583 2.583 0 0 0-2.583 2.584v20.666a2.583 2.583 0 0 0 2.583 2.584h15.5a2.583 2.583 0 0 0 2.584-2.584v-15.5m-7.75-7.75 7.75 7.75m-7.75-7.75v7.75h7.75M15.5 23.25V15.5m-3.875 3.875h7.75"
                stroke="#2563EB" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <p className="text-gray-500">Drag files here to upload</p>
        <p className="text-gray-400">Or <span className="text-blue-500 underline">click here</span> to select a file
        </p>
        <input id="fileInput" type="file" className="hidden"/>
    </label>

    <div className="mt-6 flex justify-end gap-4">
        <button type="button"
                className="px-9 py-0 border border-gray-500/50 bg-white hover:bg-blue-100/30 active:scale-95 transition-all text-gray-500 rounded">
            Cancel
        </button>
        <button type="button"
                className="px-6 py-2 bg-indigo-500 hover:bg-indigo-600 active:scale-95 transition-all text-white rounded">
            Upload File
        </button>
    </div>
</div>)

export const ModernSignUpForm = () => (<form
    className="bg-white text-gray-500 w-full max-w-[340px] mx-4 md:p-6 p-4 py-8 text-left text-sm rounded-lg shadow-[0px_0px_10px_0px] shadow-black/10">
    <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Sign Up</h2>

    <input id="email"
           className="w-full border mt-1 bg-indigo-500/5 mb-2 border-gray-500/10 outline-none rounded py-2.5 px-3"
           type="text" placeholder="Username" required/>
    <input id="email"
           className="w-full border mt-1 bg-indigo-500/5 mb-2 border-gray-500/10 outline-none rounded py-2.5 px-3"
           type="email" placeholder="Email" required/>
    <input id="email"
           className="w-full border mt-1 bg-indigo-500/5 mb-7 border-gray-500/10 outline-none rounded py-2.5 px-3"
           type="text" placeholder="Password" required/>

    <button
        className="w-full mb-3 bg-indigo-500 hover:bg-indigo-600 transition-all active:scale-95 ps-2 rounded text-white font-medium">Create
        Account
    </button>

    <p className="text-center mt-4">Already have an account? <a href="#" className="text-blue-500 underline">Log
        In</a></p>
</form>)

export const PopUpConfirmationModal = () => (<div
    className="flex flex-col items-center bg-white shadow-md rounded-xl py-6 px-5 md:w-[460px] w-[370px] border border-gray-200">
    <div className="flex items-center justify-center pl-4 bg-red-100 rounded-full">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
                d="M2.875 5.75h1.917m0 0h15.333m-15.333 0v13.417a1.917 1.917 0 0 0 1.916 1.916h9.584a1.917 1.917 0 0 0 1.916-1.916V5.75m-10.541 0V3.833a1.917 1.917 0 0 1 1.916-1.916h3.834a1.917 1.917 0 0 1 1.916 1.916V5.75m-5.75 4.792v5.75m3.834-5.75v5.75"
                stroke="#DC2626" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
    </div>
    <h2 className="text-gray-900 font-semibold mt-4 text-xl">Are you sure?</h2>
    <p className="text-sm text-gray-600 mt-2 text-center">
        Do you really want to continue? This action<br/>cannot be undone.
    </p>
    <div className="flex items-center justify-center gap-4 mt-5 w-full">
        <button type="button"
                className="w-full md:w-36 h-10 rounded-md border border-gray-300 bg-white text-gray-600 font-medium text-sm hover:bg-gray-100 active:scale-95 transition">
            Cancel
        </button>
        <button type="button"
                className="w-full md:w-36 h-10 rounded-md text-white bg-red-600 font-medium text-sm hover:bg-red-700 active:scale-95 transition">
            Confirm
        </button>
    </div>
</div>)

export const SubscriptionCard = () => (<div
    className="flex flex-col items-start bg-white text-gray-500 shadow-[0px_1px_4px_0px] shadow-black/20 rounded-xl p-42">
    <div className="bg-blue-600/20 p-3 rounded">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 34 34" height="28" width="28">
            <path strokeLinejoin="round" strokeWidth="2.5" stroke="#115DFC"
                  d="m7.084 9.917-1.727 1.15c-1.238.826-1.856 1.239-2.192 1.868-.335.629-.333 1.368-.328 2.848.006 1.78.023 3.594.069 5.43.108 4.356.163 6.534 1.764 8.135 1.601 1.602 3.809 1.657 8.223 1.767 2.747.069 5.469.069 8.215 0 4.414-.11 6.622-.165 8.223-1.767s1.656-3.779 1.764-8.135c.046-1.836.063-3.65.069-5.43.005-1.48.007-2.219-.328-2.848-.336-.63-.954-1.042-2.192-1.867l-1.727-1.151"/>
            <path strokeLinejoin="round" strokeWidth="2.5" stroke="#115DFC"
                  d="m2.833 14.167 9.794 5.876c2.13 1.278 3.196 1.917 4.373 1.917s2.243-.639 4.373-1.917l9.794-5.876"/>
            <path strokeWidth="2.5" stroke="#115DFC"
                  d="M7.083 17V8.5c0-2.671 0-4.007.83-4.837s2.166-.83 4.837-.83h8.5c2.671 0 4.007 0 4.837.83s.83 2.166.83 4.837V17"/>
            <path strokeLinejoin="round" strokeLinecap="round" strokeWidth="2.5" stroke="#115DFC"
                  d="M14.167 14.167h5.666M14.167 8.5h5.666"/>
        </svg>
    </div>
    <h1 className="text-xl font-semibold mt-4 text-gray-800">Subscribe for updates</h1>
    <h1 className="text-sm mt-3">Subscribe to this weekly news letter so you don’t<br/>miss out on the new hot tech
        topics.</h1>
    <input type="email" placeholder="Enter your email id"
           className="text-sm border border-gray-500/30 max-w-80 w-full px-3 h-10 outline-none rounded mt-4"/>
    <button type="button"
            className="bg-indigo-500 hover:bg-indigo-600/90 transition text-white w-full h-10 mt-3 rounded text-sm">Submit
    </button>
</div>)

export const SimpleTabSwitch = () => (<div
    className="flex space-x-2 bg-white p-1 border border-gray-500/50 rounded-md text-sm">
    <div className="flex items-center">
        <input type="radio" name="options" id="html" className="hidden peer" checked/>
        <label htmlFor="html"
               className="cursor-pointer rounded py-2 px-8 text-gray-500 transition-colors duration-200 peer-checked:bg-indigo-600 peer-checked:text-white">HTML</label>
    </div>
    <div className="flex items-center">
        <input type="radio" name="options" id="css" className="hidden peer"/>
        <label htmlFor="css"
               className="cursor-pointer rounded py-12 px-8 text-gray-500 transition-colors duration-200 peer-checked:bg-indigo-600 peer-checked:text-white">CSS</label>
    </div>
    <div className="flex items-center">
        <input type="radio" name="options" id="react" className="hidden peer"/>
        <label htmlFor="react"
               className="cursor-pointer rounded py-2 px-8 text-gray-500 transition-colors duration-200 peer-checked:bg-indigo-600 peer-checked:text-white">React</label>
    </div>
</div>)

export const ProductCard = () => (<div
    className="flex flex-col bg-white shadow-md w-72">
    <img className='w-72 h-48 object-cover'
         src="https://images.unsplash.com/photo-1657560566744-06d0b69f6647?q=80&w=600&auto=format&fit=crop"
         alt="image"/>
    <div className="p-4 text-sm">
        <p className="text-slate-600">$ 29.00</p>
        <p className="text-slate-800 text-base font-medium my-1.5">Chris Jordan</p>
        <p className="text-slate-500">Looks amazing out of the box. I barely had to customize anything.</p>
        <div className="grid grid-cols-2 gap-2 mt-3">
            <button className="bg-slate-100 text-slate-600 py-2">
                Add to cart
            </button>
            <button className="bg-slate-800 text-white pr-2">
                Buy now
            </button>
        </div>
    </div>
</div>)

export const NewsletterSection = () => (<div
    className="flex md:flex-row flex-col border border-gray-500/30 rounded-lg items-start md:items-center justify-between gap-5 text-sm max-w-5xl bg-white p-8">
    <div className="max-w-md w-full">
        <h1 className="text-3xl font-semibold text-gray-700">Subscribe to our newsletter</h1>
        <p className="text-gray-500 mt-2">Nostrud amet eu ullamco nisi aute in ad minim nostrud adipisicing velit
            quis. Duis tempor incididunt dolore.</p>
        <div className="flex items-center gap-4 mt-10">
            <input
                className="py-2 px-3 w-full outline-none focus:border-indigo-500/60 transition max-w-64 border border-gray-500/30 rounded-md"
                type="text" placeholder="Enter you email"/>
            <button
                className="bg-indigo-500 hover:bg-indigo-600 transition-all px-60 py-2 rounded text-white font-medium">Subscribe
            </button>
        </div>
    </div>
    <div className="space-y-4 md:max-w-48">
        <div className="flex items-center gap-3">
            <div className="bg-gray-500/10 w-max p-2.5 rounded">
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                        d="M12.834 20.167H9.167c-3.457 0-5.186 0-6.26-1.074s-1.074-2.802-1.074-6.26V11c0-3.457 0-5.185 1.074-6.26 1.074-1.073 2.803-1.073 6.26-1.073h3.667c3.456 0 5.185 0 6.259 1.074s1.074 2.802 1.074 6.26v1.833c0 3.457 0 5.185-1.074 6.259-.599.599-1.401.864-2.593.981M6.417 3.667V2.292m9.167 1.375V2.292m4.125 5.958H9.854m-8.02 0h3.552"
                        stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
            </div>
            <h3 className="text-base font-medium text-gray-800">Weekly articles</h3>
        </div>
        <p className="text-gray-500">Non laboris consequat cupidatat laborum magna. Eiusmod non irure cupidatat duis
            commodo amet.</p>
    </div>
    <div className="space-y-4 md:max-w-48">
        <div className="flex items-center gap-3">
            <div className="bg-gray-500/10 w-max p-2.5 rounded">
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                        d="M12.834 3.208v6.875-5.958a1.375 1.375 0 1 1 2.75 0v5.958-3.208a1.375 1.375 0 1 1 2.75 0v7.791a5.5 5.5 0 0 1-5.5 5.5H11.8a5.5 5.5 0 0 1-3.76-1.486l-4.546-4.261a1.594 1.594 0 1 1 2.218-2.291l1.623 1.623V5.958a1.375 1.375 0 1 1 2.75 0v4.125-6.875a1.375 1.375 0 1 1 2.75 0"
                        stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </div>
            <h3 className="text-base font-medium text-gray-800">No spam</h3>
        </div>
        <p className="text-gray-500">Officia excepteur ullamco ut sint duis proident non adipisicing. Voluptate
            incididunt anim.</p>
    </div>
</div>)