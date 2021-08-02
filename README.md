# RSSalut

Hand gesture recognition with WiFi RSS values.

## Project details
Environment developed on:
- Ubuntu 20.04 LTS

Programming language:
- ***Python*** (v3.6)

Libraries used:
- ***csv***
- ***pywt***
- ***numpy***
- ***matplotlib***
- ***scipy.signal*** 

## Quickstart
Directly start the program while providing the csv data file in the command line argument. The signal will be plotted and showed on a window with events marked. On the terminal, the timestamp of the events will be shown followed by the decoded code string and the categorized gesture type.
```
$ ./RSSalut [filename]
```

## Notes
The pcap files are not included in the moodle submission due to size limitations so only csv files for each gestures are included. The pcap files can be provided by request if needed.