D:\gstreamer\1.0\x86\bin\gst-launch-1.0.exe -vvv tcpclientsrc host=10.75.89.15 port=1180 ! gdpdepay ! queue ! avdec_vp9 ! autovideosink sync=false async=false -e