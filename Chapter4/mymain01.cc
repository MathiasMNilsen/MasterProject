#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>
using namespace Pythia8;
using namespace std;

double angle_between(Vec4 p1, Vec4 p2){
    double dot = p1.px()*p2.px() + p1.py()*p2.py() + p1.pz()*p2.pz();
    double abs_p1 = sqrt(pow(p1.px(), 2) + pow(p1.py(), 2) + pow(p1.pz(), 2));
    double abs_p2 = sqrt(pow(p2.px(), 2) + pow(p2.py(), 2) + pow(p2.pz(), 2));
    return acos( dot/(abs_p1*abs_p2) );
}

int classify_jet(Vec4 p_jet, Event pythia_event, double R){

    if ( angle_between(p_jet, pythia_event[5].p()) < R )
    {
        if (pythia_event[5].isGluon())
        {
            return 2;
        }
        else if (pythia_event[5].isQuark())
        {
            return 1;
        }
    }
    if ( angle_between(p_jet, pythia_event[6].p()) < R  )
    {
        if (pythia_event[6].isGluon())
        {
            return 2;
        }
        else if (pythia_event[6].isQuark())
        {
            return 1;
        }
    }
    return 0; 
}

int main() {    
    Pythia pythia;
    const Info& info = pythia.info;

    string e_cm = "5020";
    bool nPDF = true;
    string PDF = ""; 

    //Turn on nPDF
    if (nPDF)
    {   
        PDF = "_nPDF";
        pythia.readString("PDF:pSet = 13");
        pythia.readString("PDF:useHardNPDFB = on");
        pythia.readString("PDF:nPDFSetB = 1");
        pythia.readString("PDF:nPDFBeamB = 100822080");
        pythia.readString("PDF:useHardNPDFA = on");
        pythia.readString("PDF:nPDFSetA = 1");
        pythia.readString("PDF:nPDFBeamA = 100822080");
    }
    pythia.readString("Beams:idA = 2212");      //Proton
    pythia.readString("Beams:idB = 2212");      //Proton
    pythia.readString("Beams:eCM = " + e_cm);   //E_cm energy
    pythia.readString("HardQCD:all = on");      //pQCD
    pythia.readString("SoftQCD:all = off");
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("HadronLevel:all = off"); //No Hadronization
    pythia.readString("PhaseSpace:pTHatMin = 30.");
    pythia.readString("PhaseSpace:bias2Selection = on");
    pythia.readString("PhaseSpace:bias2SelectionPow = 4.");
    pythia.readString("Print:quiet = off");
    pythia.init();

    //Histograms
    int nBin = 25;
    double pT_min = 50;
    double pT_max = 1000;
    Hist Inclu_jet("Inclusive", nBin, pT_min, pT_max, true);
    Hist Quark_jet("Quark", nBin, pT_min, pT_max, true);
    Hist Gloun_jet("Gluon", nBin, pT_min, pT_max, true);

    //Constants
    int n_events = 1e7; //10 million events
    double R = 0.4;
    double pT_jet_min = 50;
    double eta_max = 2.8;
    string eta_max_str = "2.8";

    //Set up SlowJet jet finder, with anti-kT clustering
    SlowJet slowJet(-1, R, pT_jet_min, eta_max);
    
    //Event loop
    for (int n=0; n<n_events; ++n) {
        pythia.next();
        slowJet.analyze(pythia.event);  

        //Loop over recorded jets
        for (int i = 0; i < slowJet.sizeJet(); ++i){

            Inclu_jet.fill(slowJet.pT(i), info.weight());
            int jet_type = classify_jet(slowJet.p(i), pythia.event, R);
            if (jet_type == 1)
            {
                Quark_jet.fill(slowJet.pT(i), info.weight());
            }
            else if (jet_type == 2)
            {
                Gloun_jet.fill(slowJet.pT(i), info.weight());
            }
              
        }
        cout << slowJet.sizeJet() << endl;
    }
    pythia.stat();
    cout << 1e6*info.sigmaErr() << endl;
    Gloun_jet.normalizeSpectrum(info.weightSum()/(info.sigmaGen()*1e6));
    Quark_jet.normalizeSpectrum(info.weightSum()/(info.sigmaGen()*1e6));
    Inclu_jet.normalizeSpectrum(info.weightSum()/(info.sigmaGen()*1e6));
   
    string tag_fileName = "Data/" + e_cm 
                        + "/taggedJets_y" + eta_max_str
                        + PDF + ".csv";
    string inc_fileName = "Data/" + e_cm 
                        + "/inclusiveJets_y" + eta_max_str
                        + PDF + ".csv";

    ofstream tFile(tag_fileName);
    ofstream iFile(inc_fileName);
    table(Quark_jet, Gloun_jet, tag_fileName); 
    Inclu_jet.table(inc_fileName);
    return 0;
}