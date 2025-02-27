WORKDIR=/data/user/tvaneede/GlobalFit/EventGenerator

echo "lets go"

$WORKDIR/run_evt_gen.py \
                        --infiles /data/ana/Diffuse/GlobalFit_Flavor/taupede/SnowStorm/RecowithBfr/Baseline/22086/0000000-0000999/Reco_NuTau_000990_out.i3.bz2 \
                        --outfile /data/user/tvaneede/GlobalFit/EventGenerator/RecoEvtGen_NuTau_000990_out.i3.bz2 \
                        # --gcdfile /data/user/nlad/Ternary_Classifier/Ternary_Classifier/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz \
                        # --flavor NuTau \
                        # --icemodel Bfr \
                        # --year IC86_2011 \
                        # --innerboundary 550 \
                        # --outerboundary 650 

